#!/bin/bash
# Default configuration for SLAM-LLM ASR experiments
# Source this file before running any of the experiment scripts:
# source scripts/config_defaults.sh
#
# All variables can be overridden before sourcing this file, e.g.:
# export SPEECH_ENCODER_PATH=/path/to/your/wavlm.pt
# export PROMPT=prompt_3
# source scripts/config_defaults.sh

# ============================================
# DEFAULT VALUES
# ============================================

# Model Paths
DEFAULT_SPEECH_ENCODER_PATH=/path/to/wavlm/WavLM-Large.pt
DEFAULT_SPEECH_ENCODER_DIM=1024
DEFAULT_LLM_PATH=/path/to/vicuna/vicuna-7b-v1.5
DEFAULT_LLM_DIM=4096
# DEFAULT_LLM_PATH=/path/to/tiny-vicuna/Tiny-Vicuna-1B  # runs ok in RTX 3090
# DEFAULT_LLM_DIM=2048

# Data Paths
DEFAULT_TRAIN_DATA_PATH=/path/to/data_train.jsonl
DEFAULT_VAL_DATA_PATH=/path/to/data_dev.jsonl
DEFAULT_TEST_DATA_PATH=/path/to/data_test.jsonl

# Training Configuration
DEFAULT_NUM_EPOCHS_BASE=1
DEFAULT_NUM_EPOCHS_PP=1
DEFAULT_BATCH_SIZE=4

# Prompt and P-Projector Configuration
DEFAULT_PROMPT="prompt_1"
DEFAULT_PROMPT_METHOD="p-projector"

# Experiment Name (used for organizing outputs in exp/)
DEFAULT_EXPERIMENT_NAME="default"

# Checkpoint Configuration (by default the last checkpoint, but can be changed to best WER checkpoint)
DEFAULT_BASE_CKPT_FOLDER="epoch_1"
DEFAULT_PP_CKPT_FOLDER="epoch_1"

# System Configuration
DEFAULT_CUDA_VISIBLE_DEVICES=0
DEFAULT_TOKENIZERS_PARALLELISM=false
DEFAULT_OMP_NUM_THREADS=1

# SLURM Configuration
DEFAULT_SLURM_ACCOUNT=YOU_ACCOUNT_NAME
DEFAULT_SLURM_PARTITION=gpu
DEFAULT_SLURM_GPU_TYPE=h100
# DEFAULT_SLURM_GPU_TYPE=rtx3090
DEFAULT_SLURM_NUM_GPUS=1
DEFAULT_SLURM_TIME_TRAIN=06:00:00
DEFAULT_SLURM_TIME_DECODE=02:00:00

# ============================================
# ENVIRONMENT VARIABLE ASSIGNMENT
# (Only set if not already defined by user)
# ============================================

# Model Paths
export SPEECH_ENCODER_PATH=${SPEECH_ENCODER_PATH:-$DEFAULT_SPEECH_ENCODER_PATH}
export SPEECH_ENCODER_DIM=${SPEECH_ENCODER_DIM:-$DEFAULT_SPEECH_ENCODER_DIM}
export LLM_PATH=${LLM_PATH:-$DEFAULT_LLM_PATH}
export LLM_DIM=${LLM_DIM:-$DEFAULT_LLM_DIM}

# Data Paths
export TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$DEFAULT_TRAIN_DATA_PATH}
export VAL_DATA_PATH=${VAL_DATA_PATH:-$DEFAULT_VAL_DATA_PATH}
export TEST_DATA_PATH=${TEST_DATA_PATH:-$DEFAULT_TEST_DATA_PATH}

# Training Configuration
export NUM_EPOCHS_BASE=${NUM_EPOCHS_BASE:-$DEFAULT_NUM_EPOCHS_BASE}
export NUM_EPOCHS_PP=${NUM_EPOCHS_PP:-$DEFAULT_NUM_EPOCHS_PP}
export BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}

# Prompt and P-Projector Configuration
export PROMPT=${PROMPT:-$DEFAULT_PROMPT}
export PROMPT_METHOD=${PROMPT_METHOD:-$DEFAULT_PROMPT_METHOD}

# Experiment Name
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}

# Checkpoint Configuration
export BASE_CKPT_FOLDER=${BASE_CKPT_FOLDER:-$DEFAULT_BASE_CKPT_FOLDER}
export PP_CKPT_FOLDER=${PP_CKPT_FOLDER:-$DEFAULT_PP_CKPT_FOLDER}

# System Configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-$DEFAULT_TOKENIZERS_PARALLELISM}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$DEFAULT_OMP_NUM_THREADS}

# SLURM Configuration
export SLURM_ACCOUNT=${SLURM_ACCOUNT:-$DEFAULT_SLURM_ACCOUNT}
export SLURM_PARTITION=${SLURM_PARTITION:-$DEFAULT_SLURM_PARTITION}
export SLURM_GPU_TYPE=${SLURM_GPU_TYPE:-$DEFAULT_SLURM_GPU_TYPE}
export SLURM_NUM_GPUS=${SLURM_NUM_GPUS:-$DEFAULT_SLURM_NUM_GPUS}
export SLURM_TIME_TRAIN=${SLURM_TIME_TRAIN:-$DEFAULT_SLURM_TIME_TRAIN}
export SLURM_TIME_DECODE=${SLURM_TIME_DECODE:-$DEFAULT_SLURM_TIME_DECODE}

# ============================================
# SUMMARY
# ============================================
if [ "$RETURN_JOB_ID" != "true" ]; then
    echo "Configuration defaults loaded successfully!"
    echo "SPEECH_ENCODER_PATH: $SPEECH_ENCODER_PATH"
    echo "LLM_PATH: $LLM_PATH"
    echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
    echo "PROMPT: $PROMPT"
    echo "PROMPT_METHOD: $PROMPT_METHOD"
    echo "BASE_CKPT_FOLDER: $BASE_CKPT_FOLDER"
    echo "PP_CKPT_FOLDER: $PP_CKPT_FOLDER"
fi
