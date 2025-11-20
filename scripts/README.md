# 📂 Experiment Scripts

This directory contains scripts for training and evaluating SLAM-LLM models for Automatic Speech Recognition (ASR).

> [!NOTE]
> These scripts are designed to submit jobs to a SLURM cluster. Each script uses `sbatch` to queue training or inference jobs. See the [SLURM Configuration Variables](#slurm-configuration-variables) section for details on customizing SLURM settings for your cluster.

## 🚀 Quick Start

### 🔁 Experiments Workflow

The typical workflow for running experiments is:

1. **Configure default paths and data:**
   Edit `scripts/config_defaults.sh` to set paths for your environment:
   ```bash
   # Edit these DEFAULT_* variables in config_defaults.sh:
   DEFAULT_TRAIN_DATA_PATH=/path/to/train.jsonl
   DEFAULT_VAL_DATA_PATH=/path/to/dev.jsonl
   DEFAULT_TEST_DATA_PATH=/path/to/test.jsonl

   DEFAULT_SPEECH_ENCODER_PATH=/path/to/wavlm/WavLM-Large.pt
   DEFAULT_SPEECH_ENCODER_DIM=1024
   DEFAULT_LLM_PATH=/path/to/tiny-vicuna/Vicuna-7B
   DEFAULT_LLM_DIM=4096
   ```

2. **Select prompt configuration:**
   Choose which prompt to use (prompt_1 through prompt_8). Each prompt configuration is defined in the `conf/` folder:
   - `conf/prompt_1.yaml` - Prompt configuration 1
   - `conf/prompt_2.yaml` - Prompt configuration 2
   - ... (through prompt_8)
   
   Set the prompt via environment variable:
   ```bash
   export PROMPT=prompt_1  # or prompt_2, prompt_3, ..., prompt_8
   ```

3. **Configure SLURM settings (if needed):**
   The scripts use default SLURM settings defined in `config_defaults.sh`. If your cluster uses different account names, partitions, or GPU types, you can override them:
   ```bash
   export SLURM_ACCOUNT=your_account
   export SLURM_PARTITION=your_partition
   export SLURM_GPU_TYPE=a100  # or rtx3090, v100, h100, etc.
   ...
   ```
   See [SLURM Configuration Variables](#slurm-configuration-variables) for all available options.

4. **Run experiments with different prompts:**
   You can evaluate the same model on the same data using different prompts. This allows for prompt engineering experiments.
   
   > [!IMPORTANT]
   > The scripts must be run in order (1 (→ 2) → 3 → 4) as each depends on outputs from previous steps:
   
   ```bash
   # Example: Running complete pipeline with prompt_1
   export PROMPT=prompt_1
   
   # Step 1: Train base model (projector only)
   bash scripts/1.finetune_base.sh
   # Produces: exp/default/base/prompt_1/epoch_*/model.pt
   
   # Step 2: Evaluate base model (baseline WER)
   bash scripts/2.decode_base.sh
   # Produces: exp/default/base/prompt_1/epoch_*/decode_output
   
   # Step 3: Fine-tune with prompt projector (requires base checkpoint from step 1)
   bash scripts/3.finetune_pp.sh
   # Produces: exp/default/p-projector/prompt_1/epoch_*/
   
   # Step 4: Evaluate prompt projector model (final WER)
   bash scripts/4.decode_pp.sh
   # Produces: exp/default/p-projector/prompt_1/epoch_*/decode_output
   ```

   **Goal:** Compare baseline WER (from step 2) vs. final WER after prompt projector training (from step 4).
   WER is saved as "WER.txt" file inside the targeted checkpoint folders ($DEFAULT_BASE_CKPT_FOLDER, $DEFAULT_PP_CKPT_FOLDER) inside the `exp/` folder

   > [!TIP]
   > You can run all these steps automatically in sequence using the master script:
   > ```bash
   > # Run the entire pipeline (steps 1-4) with proper SLURM job dependencies
   > export PROMPT=prompt_1
   > bash scripts/run_all.sh
   > ```
   > This will submit all jobs to SLURM where Step 2 and 3 wait for Step 1, and Step 4 waits for Step 3. See the [Scripts Overview](#scripts-overview) section for more details.


### ✅ Complete Workflow

1. **Set up environment variables:**
   ```bash
   source scripts/config_defaults.sh
   ```

2. **Train base model (projector only):**
   ```bash
   bash scripts/1.finetune_base.sh
   ```
   This will create the base checkpoints at: `exp/$EXPERIMENT_NAME/base/$PROMPT/epoch_*/`

3. **Evaluate base model:**
   ```bash
   bash scripts/2.decode_base.sh
   ```

4. **Fine-tune with prompt projector method:**
   ```bash
   bash scripts/3.finetune_pp.sh
   ```
   This will create the final checkpoints at: `exp/$EXPERIMENT_NAME/$PROMPT_METHOD/$PROMPT/epoch_*/`

5. **Evaluate prompt projector model:**
   ```bash
   bash scripts/4.decode_pp.sh
   ```

### ⚙️ Running with Different Configurations

You can override environment variables for individual runs:

#### Use a different experiment name:
```bash
EXPERIMENT_NAME=my_experiment bash scripts/1.finetune_base.sh
```

#### Use a different prompt:
```bash
PROMPT=prompt_2 bash scripts/1.finetune_base.sh
```

#### Use p-tuning instead of p-projector:
```bash
PROMPT_METHOD=p-tuning bash scripts/3.finetune_pp.sh
```

#### Use different SLURM settings:
```bash
SLURM_GPU_TYPE=a100 SLURM_NUM_GPUS=2 bash scripts/1.finetune_base.sh
```

#### Use a specific checkpoint for decoding:
```bash
# Decode using a different base checkpoint
BASE_CKPT_FOLDER=epoch_3 bash scripts/2.decode_base.sh

# Decode using a different prompt projector checkpoint
PP_CKPT_FOLDER=epoch_2 bash scripts/4.decode_pp.sh
```

#### Combine multiple overrides:
```bash
EXPERIMENT_NAME=llama2 PROMPT=prompt_3 BASE_CKPT_FOLDER=epoch_4 bash scripts/3.finetune_pp.sh
```

## 🔗 Experiment Pipeline Dependencies

The scripts have the following dependencies:

```
1.finetune_base.sh
|   ↓
|    (produces checkpoint: exp/$EXPERIMENT_NAME/base/$PROMPT/epoch_X/)
|    ↓
|   2.decode_base.sh ← uses base checkpoint
↓
3.finetune_pp.sh ← requires base checkpoint
    ↓
    (produces checkpoint: exp/$EXPERIMENT_NAME/$PROMPT_METHOD/$PROMPT/epoch_X/)
    ↓
4.decode_pp.sh ← uses both base and prompt projector checkpoints
```

## 📜 Scripts Overview

### Master Script
- **run_all.sh** - Submits all 4 jobs automatically with SLURM dependencies (Step 2 and 3 depend on Step 1; Step 4 depends on Step 3)

### Training Scripts
1. **1.finetune_base.sh** - Fine-tunes the base model (projector layer only)
2. **3.finetune_pp.sh** - Fine-tunes with prompt projector (p-tuning or p-projector)

### Inference Scripts
3. **2.decode_base.sh** - Decodes/evaluates the base fine-tuned model
4. **4.decode_pp.sh** - Decodes/evaluates the prompt projector fine-tuned model

## 🗄️ Output Directories

All experiments create outputs under `exp/$EXPERIMENT_NAME/` in the project root folder:

- **Base model:** `exp/$EXPERIMENT_NAME/base/$PROMPT/epoch_*/`
  - `model.pt` - Base model checkpoint
  - `train.*.out` - Training stdout log
  - `train.*.err` - Training stderr log
  - `decode_output` - Decoding results (after running script 2)

- **Prompt projector model:** `exp/$EXPERIMENT_NAME/$PROMPT_METHOD/$PROMPT/epoch_*/`
  - Model checkpoints
  - Training logs
  - `decode_output` - Decoding results (after running script 4)

> [!TIP]
> Use `EXPERIMENT_NAME` to organize different experiments. For example, when testing different LLMs or datasets:
> ```bash
> EXPERIMENT_NAME=vicuna-7b bash scripts/run_all.sh
> EXPERIMENT_NAME=llama2-7b bash scripts/run_all.sh
> EXPERIMENT_NAME=my_custom_data bash scripts/run_all.sh
> ```

## 🧩 Configuration

### 🧱 Setting Up Environment Variables

Before running any scripts, you must set up the environment variables.

All scripts depend on common environment variables defined in `config_defaults.sh` by running .

> [!IMPORTANT]
> The `config_defaults.sh` script respects user-defined variables. It will only set default values for variables that are not already defined. This means you can override any variable before sourcing the file.

#### Option 1: Source the environment file (Recommended)
```bash
# From the scripts directory or project root
source scripts/config_defaults.sh
```

#### Option 2: Override specific variables before sourcing
```bash
# Set custom values for specific variables
export SPEECH_ENCODER_PATH=/path/to/your/wavlm.pt
export LLM_PATH=/path/to/your/llm
export PROMPT=prompt_3
export BATCH_SIZE=8

# Then source to set defaults for remaining variables
source scripts/config_defaults.sh
```

#### Option 3: Override per-script execution
```bash
# Override for a single script run without changing environment
PROMPT=prompt_2 BATCH_SIZE=16 bash scripts/1.finetune_base.sh
```

#### Option 4: Modify default values in config_defaults.sh

Edit the `DEFAULT_*` variables at the top of `scripts/config_defaults.sh`:
```bash
# Edit the file to change default values
vim scripts/config_defaults.sh

# Then source it
source scripts/config_defaults.sh
```

## 📋 Environment Variables Reference

### 🗂️ Model and Data Path Variables

| Variable | Description |
|----------|-------------|
| `SPEECH_ENCODER_PATH` | Path to WavLM encoder checkpoint |
| `LLM_PATH` | Path to LLM model directory |
| `TRAIN_DATA_PATH` | Training data JSONL file path |
| `VAL_DATA_PATH` | Validation data JSONL file path |
| `TEST_DATA_PATH` | Test data JSONL file path |

> [!NOTE]
> Default values are set in `scripts/config_defaults.sh`. Modify the `DEFAULT_*` variables in that file to match your setup.

### 🧬 Model Configuration Variables

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `SPEECH_ENCODER_DIM` | `1024` | Dimension of speech encoder output |
| `LLM_DIM` | `2048` | Dimension of LLM embeddings |

### 🏋️ Training Configuration Variables

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `NUM_EPOCHS_BASE` | `1` | Number of epochs for base training |
| `NUM_EPOCHS_PP` | `1` | Number of epochs for prompt projector training |
| `BATCH_SIZE` | `4` | Batch size |

### 🧪 Experiment Configuration Variables

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `EXPERIMENT_NAME` | `default` | Name for organizing experiment outputs (e.g., `exp/$EXPERIMENT_NAME/base/...`) |
| `PROMPT` | `prompt_1` | Prompt configuration to use (prompt_1 through prompt_8) |
| `PROMPT_METHOD` | `p-projector` | Prompt projector method: `p-tuning` or `p-projector` |

### 🪣 Checkpoint Configuration Variables

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `BASE_CKPT_FOLDER` | `epoch_1` | Folder name of the base model checkpoint to use |
| `PP_CKPT_FOLDER` | `epoch_1` | Folder name of the prompt projector model checkpoint to use |

### 💻 System Configuration Variables

| Variable | Default Value | Description |
|----------|-------------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device(s) to use |
| `TOKENIZERS_PARALLELISM` | `false` | Disable tokenizer parallelism warnings |
| `OMP_NUM_THREADS` | `1` | Number of OpenMP threads |

### 🖥️ SLURM Configuration Variables

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `SLURM_ACCOUNT` | - | SLURM account name |
| `SLURM_PARTITION` | `gpu` | SLURM partition to use |
| `SLURM_GPU_TYPE` | `h100` | GPU type to request (Tiny-Vicuna-1B runs OK on RTX3090) |
| `SLURM_NUM_GPUS` | `1` | Number of GPUs to request |
| `SLURM_TIME_TRAIN` | `06:00:00` | Time limit for training jobs |
| `SLURM_TIME_DECODE` | `02:00:00` | Time limit for decoding jobs |

## 🩺 Troubleshooting

### 🔍 Monitoring SLURM jobs from run_all.sh

When you run `run_all.sh`, it will display commands to monitor your jobs:

```bash
# Check status of all your jobs
squeue -u $USER

# Check specific jobs from the pipeline
squeue -j JOB1_ID,JOB2_ID,JOB3_ID,JOB4_ID

# Detailed job information
sacct -j JOB1_ID,JOB2_ID,JOB3_ID,JOB4_ID --format=JobID,JobName,State,ExitCode,Elapsed

# Cancel all jobs in the pipeline (if needed)
scancel JOB1_ID JOB2_ID JOB3_ID JOB4_ID
```

> [!NOTE]
> If a job fails, dependent jobs will not run (they will remain in the queue with status `DependencyNeverSatisfied`).

### 🚫 SLURM job submission fails
Check your SLURM configuration variables match your cluster setup:
```bash
echo $SLURM_ACCOUNT
echo $SLURM_PARTITION
echo $SLURM_GPU_TYPE
```

### 📌 Checkpoint paths don't match
The decode and prompt projector training scripts use checkpoint paths specified by environment variables. If your checkpoint has a different folder name, set the appropriate variable:

```bash
# Find your checkpoint folder name
ls exp/$EXPERIMENT_NAME/base/prompt_1/

# Use it for decoding
BASE_CKPT_FOLDER=epoch_4 bash scripts/2.decode_base.sh

# Or for prompt projector training
BASE_CKPT_FOLDER=epoch_4 bash scripts/3.finetune_pp.sh
```

You can also edit the default values in `scripts/config_defaults.sh` if you want to use different checkpoints consistently.

## 📝 Notes

- All scripts use SLURM's `sbatch` for job submission
- Scripts use PyTorch's `torchrun` for distributed training
- The base training freezes the encoder and LLM, only training the projector
- Prompt projector training freezes everything except the prompt projector parameters
- Batch size and number of epochs can be adjusted via environment variables
