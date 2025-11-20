#!/bin/bash
# Master script to run all training and decoding steps with SLURM job dependencies
# This script submits all jobs in sequence, where each job waits for the previous one to complete

# Determine script and run directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(dirname "$SCRIPT_DIR")"

# Source environment variables
source "$SCRIPT_DIR/config_defaults.sh"

cd "$SCRIPT_DIR"

echo "==============================================="
echo "Starting SLAM-LLM ASR Training Pipeline"
echo "==============================================="
echo "Configuration:"
echo "  PROMPT: $PROMPT"
echo "  PROMPT_METHOD: $PROMPT_METHOD"
echo "  SLURM_ACCOUNT: $SLURM_ACCOUNT"
echo "  SLURM_PARTITION: $SLURM_PARTITION"
echo "==============================================="
echo ""

# Enable job ID return mode
export RETURN_JOB_ID=true

# Step 1: Fine-tune base model (projector training)
echo "[Step 1/4] Submitting base model fine-tuning job..."
job1_id=$(bash "$SCRIPT_DIR/1.finetune_base.sh")
if [ -z "$job1_id" ]; then
    echo "ERROR: Failed to submit Step 1 job"
    exit 1
fi
echo "  Job ID: $job1_id"
echo ""

# Step 2: Decode base model (depends on Step 1)
echo "[Step 2/4] Submitting base model decoding job (depends on job $job1_id)..."
export DEPENDENCY_JOB_ID=$job1_id
job2_id=$(bash "$SCRIPT_DIR/2.decode_base.sh")
if [ -z "$job2_id" ]; then
    echo "ERROR: Failed to submit Step 2 job"
    exit 1
fi
echo "  Job ID: $job2_id"
echo ""

# Step 3: Fine-tune with prompt/p-projector (depends on Step 1)
echo "[Step 3/4] Submitting prompt/p-projector fine-tuning job (depends on job $job1_id)..."
export DEPENDENCY_JOB_ID=$job1_id
job3_id=$(bash "$SCRIPT_DIR/3.finetune_pp.sh")
if [ -z "$job3_id" ]; then
    echo "ERROR: Failed to submit Step 3 job"
    exit 1
fi
echo "  Job ID: $job3_id"
echo ""

# Step 4: Decode with prompt/p-projector (depends on Step 3)
echo "[Step 4/4] Submitting prompt/p-projector decoding job (depends on job $job3_id)..."
export DEPENDENCY_JOB_ID=$job3_id
job4_id=$(bash "$SCRIPT_DIR/4.decode_pp.sh")
if [ -z "$job4_id" ]; then
    echo "ERROR: Failed to submit Step 4 job"
    exit 1
fi
echo "  Job ID: $job4_id"
echo ""

# Summary
echo "==============================================="
echo "All jobs submitted successfully!"
echo "==============================================="
echo "Pipeline job chain:"
echo "  Step 1 (Base training):    Job $job1_id"
echo "  Step 2 (Base decoding):    Job $job2_id (waits for $job1_id)"
echo "  Step 3 (PP training):      Job $job3_id (waits for $job1_id)"
echo "  Step 4 (PP decoding):      Job $job4_id (waits for $job3_id)"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j $job1_id,$job2_id,$job3_id,$job4_id"
echo ""
echo "Check job status:"
echo "  sacct -j $job1_id,$job2_id,$job3_id,$job4_id --format=JobID,JobName,State,ExitCode,Elapsed"
echo ""
echo "Cancel all jobs:"
echo "  scancel $job1_id $job2_id $job3_id $job4_id"
echo "==============================================="
