import argparse
import json
import inspect  # Added for Transformers version compatibility
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import torch
import yaml
from datasets import load_dataset, DatasetDict
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)


# --------------------------
# Helpers
# --------------------------

def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unknown torch_dtype: {s}")

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def _safe_exp(x: float) -> float:
    x = min(float(x), 50.0)
    return float(math.exp(x))

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _looks_like_model_dir(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    if (p / "config.json").exists():
        return True
    if any(p.glob("*.safetensors")) or any(p.glob("pytorch_model*.bin")):
        return True
    return False

def _detect_text_field(example: Dict[str, Any]) -> Optional[str]:
    for k, v in example.items():
        if isinstance(v, str) and v.strip():
            return k
    return None

def _infer_target_modules(model) -> List[str]:
    names = set()
    for n, _ in model.named_modules():
        names.add(n.split(".")[-1])

    for group in [
        ["q_proj", "k_proj", "v_proj", "o_proj"],
        ["Wqkv", "out_proj"],
        ["query_key_value", "dense"],
        ["c_attn", "c_proj"],
    ]:
        if all(x in names for x in group):
            return group

    fallback = [x for x in ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "out_proj", "dense"] if x in names]
    if fallback:
        return fallback

    raise ValueError("Could not auto-infer target_modules. Set peft.target_modules explicitly.")

def _choose_attn_impl(cfg: Dict[str, Any]) -> Optional[str]:
    return cfg.get("model", {}).get("attn_implementation", None)


# --------------------------
# JSONL Logger Callback
# --------------------------

class JsonlLoggerCallback(TrainerCallback):
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.train_log_path = _ensure_dir(run_dir / "logs") / "train.jsonl"
        self.eval_log_path = _ensure_dir(run_dir / "logs") / "eval.jsonl"
        self.start_time = None

    def _eta(self, global_step: int, max_steps: int) -> Optional[str]:
        if self.start_time is None or global_step <= 0 or max_steps <= 0:
            return None
        elapsed = time.time() - self.start_time
        sec_per_step = elapsed / global_step
        remaining = max(0, max_steps - global_step) * sec_per_step
        h = int(remaining // 3600)
        m = int((remaining % 3600) // 60)
        s = int(remaining % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        max_steps = int(state.max_steps) if getattr(state, "max_steps", None) else 0
        progress_pct = (100.0 * state.global_step / max_steps) if max_steps > 0 else None
        epoch_pct = None
        if state.epoch is not None and args.num_train_epochs and args.num_train_epochs > 0:
            epoch_pct = 100.0 * (float(state.epoch) / float(args.num_train_epochs))

        payload = {
            "ts": _now_iso(),
            "event": "train_log",
            "step": int(state.global_step),
            "epoch": round(float(state.epoch), 4) if state.epoch is not None else None,
            "progress_pct": round(progress_pct, 2) if progress_pct is not None else None,
            "epoch_pct": round(epoch_pct, 2) if epoch_pct is not None else None,
            "eta": self._eta(int(state.global_step), max_steps),
            "max_grad_norm": getattr(args, "max_grad_norm", None),
            **logs,
        }

        with self.train_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        eval_loss = metrics.get("eval_loss", None)
        ppl = _safe_exp(eval_loss) if eval_loss is not None else None

        payload = {
            "ts": _now_iso(),
            "event": "eval",
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            **metrics,
            "perplexity": ppl,
        }
        with self.eval_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# --------------------------
# Data Pipeline (EOS + Packing)
# --------------------------

def build_datasets(cfg: Dict[str, Any], tokenizer) -> Tuple[Any, Any]:
    data_cfg = cfg["data"]
    train_path = data_cfg["train_jsonl"]
    eval_path = data_cfg.get("eval_jsonl", None)
    split_ratio = float(data_cfg.get("eval_split_ratio", 0.0))
    text_field = data_cfg.get("text_field", "text")
    block_size = int(data_cfg.get("block_size", 2048))
    shuffle = bool(data_cfg.get("shuffle", True))
    num_proc = int(data_cfg.get("num_proc", 4))

    pack_mode = str(data_cfg.get("pack_mode", "drop")).lower().strip()
    if pack_mode not in ("drop", "pad"):
        raise ValueError(f"data.pack_mode must be 'drop' or 'pad', got: {pack_mode}")

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id; CPT packing needs an EOS delimiter.")

    if tokenizer.pad_token_id is None:
        # safe default for many causal LMs
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    ds = load_dataset("json", data_files={"train": train_path})

    if eval_path:
        ds_eval = load_dataset("json", data_files={"eval": eval_path})
        dsd = DatasetDict({"train": ds["train"], "eval": ds_eval["eval"]})
    else:
        if 0.0 < split_ratio < 1.0:
            split = ds["train"].train_test_split(test_size=split_ratio, seed=int(cfg["run"].get("seed", 42)))
            dsd = DatasetDict({"train": split["train"], "eval": split["test"]})
        else:
            dsd = DatasetDict({"train": ds["train"], "eval": None})

    if text_field not in dsd["train"].column_names:
        auto_field = _detect_text_field(dsd["train"][0])
        if not auto_field:
            raise ValueError(f"Could not find text field. Columns: {dsd['train'].column_names}")
        text_field = auto_field

    def tokenize_fn(examples):
        out = tokenizer(
            examples[text_field],
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        # Add EOS between docs
        out["input_ids"] = [ids + [eos_id] for ids in out["input_ids"]]
        out["attention_mask"] = [m + [1] for m in out["attention_mask"]]
        return out

    tokenized_train = dsd["train"].map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=dsd["train"].column_names,
        desc="Tokenizing train",
    )

    tokenized_eval = None
    if dsd["eval"] is not None:
        tokenized_eval = dsd["eval"].map(
            tokenize_fn,
            batched=True,
            num_proc=num_proc,
            remove_columns=dsd["eval"].column_names,
            desc="Tokenizing eval",
        )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])

        if total_length == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        full_len = (total_length // block_size) * block_size
        blocks_input, blocks_attn, blocks_labels = [], [], []

        # full blocks
        for i in range(0, full_len, block_size):
            chunk = concatenated["input_ids"][i:i + block_size]
            attn = concatenated["attention_mask"][i:i + block_size]
            blocks_input.append(chunk)
            blocks_attn.append(attn)
            blocks_labels.append(chunk.copy())

        # remainder
        remainder = total_length - full_len
        if remainder > 0 and pack_mode == "pad":
            chunk = concatenated["input_ids"][full_len:full_len + remainder]
            attn = concatenated["attention_mask"][full_len:full_len + remainder]

            pad_len = block_size - remainder
            chunk_padded = chunk + [pad_id] * pad_len
            attn_padded = attn + [0] * pad_len

            labels = chunk_padded.copy()
            labels[-pad_len:] = [-100] * pad_len  # loss mask

            blocks_input.append(chunk_padded)
            blocks_attn.append(attn_padded)
            blocks_labels.append(labels)

        return {
            "input_ids": blocks_input,
            "attention_mask": blocks_attn,
            "labels": blocks_labels,
        }

    tokenized_train = tokenized_train.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc=f"Packing train blocks (mode={pack_mode})",
    )
    if tokenized_eval is not None:
        tokenized_eval = tokenized_eval.map(
            group_texts,
            batched=True,
            num_proc=num_proc,
            desc=f"Packing eval blocks (mode={pack_mode})",
        )

    if len(tokenized_train) == 0:
        raise ValueError(
            "Train dataset is empty after packing. "
            "Either increase data, reduce block_size, or set data.pack_mode='pad'."
        )

    if shuffle:
        tokenized_train = tokenized_train.shuffle(seed=int(cfg["run"].get("seed", 42)))

    return tokenized_train, tokenized_eval


# --------------------------
# Model Loading + PEFT
# --------------------------

def load_base_model_and_tokenizer(cfg: Dict[str, Any], base_dir: Path):
    model_cfg = cfg["model"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    use_fast = bool(model_cfg.get("tokenizer_use_fast", True))
    device_map = model_cfg.get("device_map", "auto")

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_dir),
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = _dtype_from_str(model_cfg.get("torch_dtype", "bfloat16"))
    use_4bit = bool(model_cfg.get("use_4bit", False))

    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(model_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=_dtype_from_str(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        )

    attn_impl = _choose_attn_impl(cfg)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(base_dir),
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
            torch_dtype=(torch_dtype if not use_4bit else None),
            quantization_config=quant_cfg,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        if attn_impl is not None:
            print(f"[warn] attn_implementation='{attn_impl}' failed: {e}")
            print("[warn] Falling back to default attention implementation.")
        model = AutoModelForCausalLM.from_pretrained(
            str(base_dir),
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
            torch_dtype=(torch_dtype if not use_4bit else None),
            quantization_config=quant_cfg,
        )

    return model, tokenizer


def apply_peft(cfg: Dict[str, Any], model):
    peft_cfg = cfg["peft"]
    model_cfg = cfg["model"]
    tr_cfg = cfg["train"]

    if not bool(peft_cfg.get("enabled", True)):
        return model, None

    use_4bit = bool(model_cfg.get("use_4bit", False))
    gradient_checkpointing = bool(tr_cfg.get("gradient_checkpointing", True))

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )

    target_modules = peft_cfg.get("target_modules", "auto")
    if target_modules == "auto":
        target_modules = _infer_target_modules(model)

    lora_config = LoraConfig(
        r=int(peft_cfg.get("r", 16)),
        lora_alpha=int(peft_cfg.get("lora_alpha", 32)),
        lora_dropout=float(peft_cfg.get("lora_dropout", 0.05)),
        bias=str(peft_cfg.get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    return model, lora_config


# --------------------------
# Merge Logic
# --------------------------

def merge_adapter(cfg: Dict[str, Any], base_dir: Path, adapter_dir: Path, final_dir: Path):
    print(f"--- Merge: {adapter_dir} + {base_dir} -> {final_dir} ---")

    model_cfg = cfg["model"]
    merge_cfg = cfg.get("merge", {})
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))

    merged_dtype = _dtype_from_str(merge_cfg.get("merged_dtype", "float16"))
    max_shard_size = str(merge_cfg.get("max_shard_size", "2GB"))

    base = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        torch_dtype=merged_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )

    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = merged.merge_and_unload()

    _ensure_dir(final_dir)
    merged.save_pretrained(str(final_dir), safe_serialization=True, max_shard_size=max_shard_size)

    tok = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(str(final_dir))

    print("--- Merge complete ---")


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--merge-only", action="store_true", help="Skip training, just merge adapter")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_dir = _ensure_dir(Path(cfg["run"]["run_dir"]))
    _ensure_dir(run_dir / "logs")

    with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    model_cfg = cfg["model"]
    repo_id = str(model_cfg["repo_id"]).strip()
    repo_path = Path(repo_id)

    # ✅ Local model path -> load directly; no download
    if repo_path.exists() and repo_path.is_dir():
        base_dir = repo_path
        if not _looks_like_model_dir(base_dir):
            raise ValueError(f"model.repo_id points to a directory, but it doesn't look like a HF model dir: {base_dir}")
    else:
        # HF repo_id -> download into run_dir/base_local_dir
        base_dir = _ensure_dir(run_dir / model_cfg.get("base_local_dir", "base_model"))
        if not _looks_like_model_dir(base_dir):
            print(f"Base model not found at {base_dir}, downloading from {repo_id} ...")
            snapshot_download(
                repo_id=repo_id,
                revision=model_cfg.get("revision", None),
                local_dir=str(base_dir),
                local_dir_use_symlinks=False,
            )

    ckpt_dir = _ensure_dir(run_dir / "checkpoints")
    best_adapter_dir = _ensure_dir(run_dir / "best_adapter")

    merge_cfg = cfg.get("merge", {}) or {}
    if merge_cfg.get("output_dir"):
        od = Path(str(merge_cfg["output_dir"]))
        final_dir = od if od.is_absolute() else (run_dir / od)
    else:
        final_dir = run_dir / "final_model"

    # Merge-only
    if args.merge_only:
        if not _looks_like_model_dir(best_adapter_dir):
            raise FileNotFoundError(f"Adapter not found at {best_adapter_dir}")
        merge_adapter(cfg, base_dir, best_adapter_dir, final_dir)
        return

    # Training
    set_seed(int(cfg["run"].get("seed", 42)))

    model, tokenizer = load_base_model_and_tokenizer(cfg, base_dir)
    model, _ = apply_peft(cfg, model)

    train_ds, eval_ds = build_datasets(cfg, tokenizer)

    tr_cfg = cfg["train"]

    dtype = _dtype_from_str(model_cfg.get("torch_dtype", "bfloat16"))
    use_fp16 = (dtype == torch.float16)
    use_bf16 = (dtype == torch.bfloat16)

    max_steps = int(tr_cfg.get("max_steps", 0))
    num_train_epochs = float(tr_cfg.get("num_train_epochs", 1))

    # --- Dynamic evaluation strategy parameter handling ---
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

    ta_kwargs = dict(
        output_dir=str(ckpt_dir),
        overwrite_output_dir=False,

        max_steps=max_steps if max_steps > 0 else -1,
        num_train_epochs=num_train_epochs,

        per_device_train_batch_size=int(tr_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(tr_cfg.get("per_device_eval_batch_size", tr_cfg.get("per_device_train_batch_size", 1))),
        gradient_accumulation_steps=int(tr_cfg.get("gradient_accumulation_steps", 1)),

        learning_rate=float(tr_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0)),
        warmup_ratio=float(tr_cfg.get("warmup_ratio", 0.0)),
        lr_scheduler_type=str(tr_cfg.get("lr_scheduler_type", "cosine")),

        optim=str(tr_cfg.get("optim", "paged_adamw_8bit" if bool(model_cfg.get("use_4bit", False)) else "adamw_torch")),
        max_grad_norm=float(tr_cfg.get("max_grad_norm", 1.0)),

        logging_steps=int(tr_cfg.get("logging_steps", 10)),

        save_strategy=str(tr_cfg.get("save_strategy", "steps")),
        save_steps=int(tr_cfg.get("save_steps", 200)),
        save_total_limit=int(tr_cfg.get("save_total_limit", 3)),

        eval_steps=int(tr_cfg.get("eval_steps", 200)),

        load_best_model_at_end=bool(tr_cfg.get("load_best_model_at_end", True)) if eval_ds is not None else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=use_fp16,
        bf16=use_bf16,

        report_to=[],
        remove_unused_columns=False,
        save_safetensors=True,
    )

    # Set the correct argument name for this transformers version
    ta_kwargs[eval_key] = str(tr_cfg.get("evaluation_strategy", "steps" if eval_ds is not None else "no"))

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[JsonlLoggerCallback(run_dir)],
    )

    # Resume
    resume_from = tr_cfg.get("resume_from_checkpoint", None)
    if resume_from == "auto":
        last = get_last_checkpoint(str(ckpt_dir))
        resume_from = last if last else None
        if resume_from:
            print(f"Resuming from {resume_from}")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from)

    trainer.save_model(str(best_adapter_dir))
    print(f"Saved best adapter -> {best_adapter_dir}")

    if eval_ds is not None:
        metrics = trainer.evaluate()
        eval_loss = metrics.get("eval_loss", None)
        metrics["perplexity"] = _safe_exp(eval_loss) if eval_loss is not None else None
        with (run_dir / "eval_final.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Final eval_loss={eval_loss}, ppl={metrics['perplexity']}")

    if bool(cfg.get("merge", {}).get("enabled", False)):
        del trainer, model
        torch.cuda.empty_cache()
        merge_adapter(cfg, base_dir, best_adapter_dir, final_dir)
    else:
        print("Merge disabled. Run with --merge-only later if needed.")


if __name__ == "__main__":
    main()
