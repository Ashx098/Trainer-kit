# Trainer‑Kit : Config‑Driven CPT (LoRA / QLoRA) with Packing, Logging, Resume, and Merge

Trainer‑Kit is a small, config‑driven training runner for **continued pretraining (CPT)** on causal LMs.
It supports **LoRA** and **QLoRA**, data **packing** (strict or padding‑masked), **checkpointing + resume**, **JSONL logging**, periodic **eval with perplexity**, and an optional **merge** step to export a final merged model.

---

## What we built

### ✅ Core goals implemented

* **CPT training loop** controlled entirely via a **YAML config**
* **Local model support** (load from filesystem) and optional **HF download** (if `repo_id` is a hub id)
* **JSONL datasets** for train (+ optional eval split)
* **CPT‑style token stream packing** into fixed‑length blocks
* **Two packing modes**

  * `drop`: strict CPT, drop remainder tokens (preferred for real CPT)
  * `pad`: pad the remainder to `block_size` and **mask loss** on padding (useful for small datasets / debugging)
* **Checkpointing + resume**

  * `resume_from_checkpoint: "auto"` resumes from the latest checkpoint under `run_dir/checkpoints`
* **JSONL logs** written locally

  * training logs: `run_dir/logs/train.jsonl`
  * eval logs: `run_dir/logs/eval.jsonl`
* **Evaluation**

  * logs `eval_loss` and computed `perplexity = exp(eval_loss)` (with safe overflow guard)
* **Adapter output**

  * saves the final/best adapter to `run_dir/best_adapter`
* **Merge workflow**

  * `--merge-only` merges an existing adapter later
  * merge is done **on CPU** to avoid GPU OOM
  * merged model is stored under the configured merge output directory (relative to `run_dir` if a relative path)

---

## Repository layout (outputs)

A run produces the following structure under `run.run_dir`:

```
runs/<run_name>/
├─ checkpoints/            # trainer checkpoints (for resume)
├─ best_adapter/           # saved LoRA adapter
├─ logs/
│  ├─ train.jsonl          # step-wise training logs
│  └─ eval.jsonl           # eval logs (eval_loss + perplexity)
├─ eval_final.json         # final eval metrics summary (if eval is enabled)
└─ config_resolved.yaml    # exact config used for the run
```

If merge is used, the merged model is written to:

* `run_dir/<merge.output_dir>` if `merge.output_dir` is relative (e.g. `./merged_model`)
* or the absolute path if it is absolute.

---

## Supported training modes

### 1) LoRA vs QLoRA (same script)

* **QLoRA** happens when `model.use_4bit: true`

  * base weights are loaded in 4‑bit using bitsandbytes
  * training updates only LoRA parameters
* **LoRA** happens when `model.use_4bit: false`

  * base weights are loaded in fp16/bf16 (as configured)
  * training updates only LoRA parameters

No “full finetune” mode is enabled by default in this runner.

---

## Data pipeline (CPT behavior)

### Input format

* JSONL file where each line contains a text field (default `"text"`).
* Example:

  * `{"text": "some training text..."}`

### Packing (token stream → fixed blocks)

* Each sample is tokenized without truncation.
* An **EOS token is appended** per document to preserve boundaries.
* Token lists are concatenated and converted into **fixed‑length blocks** of `data.block_size`.

Two modes:

* **`drop` (strict CPT):** remainder tokens that don’t fill a full block are discarded.
* **`pad` (debug/small data):** remainder is padded to block_size:

  * `attention_mask = 0` for padded positions
  * `labels = -100` for padded positions (loss masking)

This is what allowed training to proceed even with tiny dummy datasets at `block_size=1024`.

---

## Logging

Trainer‑Kit writes **machine‑readable logs** in JSONL.

### Training logs (`logs/train.jsonl`)

Includes entries with:

* `step`
* `loss`
* `grad_norm`
* `learning_rate`
* `progress_pct` (step progress when `max_steps` is active)
* ETA estimation

### Eval logs (`logs/eval.jsonl`)

Includes:

* `eval_loss`
* `perplexity`

Notes:

* When using `max_steps`, the Trainer’s internal `epoch` counter can grow unexpectedly on tiny datasets (because steps/epoch becomes ~1).
  **Use `progress_pct` as the reliable indicator** for step‑based runs.

---

## Checkpointing and resume

The trainer saves checkpoints under:

* `run_dir/checkpoints/`

Resume options:

* `resume_from_checkpoint: "auto"` → picks the latest checkpoint automatically
* `resume_from_checkpoint: "/path/to/checkpoint"` → resumes from a specific checkpoint
* `resume_from_checkpoint: null` → fresh run

---

## Merging adapters into a final model

Trainer‑Kit supports exporting a merged model:

### Merge after training

* Enable merge in config (`merge.enabled: true`)
* The script will:

  1. save the adapter
  2. free GPU memory
  3. reload base model on **CPU**
  4. load adapter
  5. `merge_and_unload()`
  6. save final merged model

### Merge later

Run:

```
python run_cpt.py --config config.yaml --merge-only
```

This skips training and merges `run_dir/best_adapter` into the base model.

---

## How to run

### Train

```
python run_cpt.py --config config.yaml
```

### Merge only

```
python run_cpt.py --config config.yaml --merge-only
