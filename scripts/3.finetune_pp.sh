#!/bin/bash
# Determine script and run directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(dirname "$SCRIPT_DIR")"

# Source environment variables
source "$SCRIPT_DIR/config_defaults.sh"

cd $RUN_DIR

# Script-specific configuration
projector_ckpt_path=$RUN_DIR/exp/$EXPERIMENT_NAME/base/${PROMPT}/${BASE_CKPT_FOLDER}
output_dir=$RUN_DIR/exp/$EXPERIMENT_NAME/${PROMPT_METHOD}/${PROMPT}

[ ! -d $output_dir ] && mkdir -p $output_dir

hydra_args="hydra.run.dir=$output_dir \
++model_config.llm_name=vicuna-1b \
++model_config.llm_path=$LLM_PATH \
++model_config.llm_dim=$LLM_DIM \
++model_config.encoder_name=wavlm \
++model_config.normalize=true \
++dataset_config.normalize=true \
++model_config.encoder_path=$SPEECH_ENCODER_PATH \
++model_config.encoder_dim=$SPEECH_ENCODER_DIM \
++model_config.encoder_projector=linear \
++model_config.encoder_projector_ds_rate=5 \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$TRAIN_DATA_PATH \
++dataset_config.val_data_path=$VAL_DATA_PATH \
++dataset_config.input_type=raw \
++train_config.model_name=asr \
++train_config.num_epochs=$NUM_EPOCHS_PP \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.freeze_projector=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=$BATCH_SIZE \
++train_config.val_batch_size=$BATCH_SIZE \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++train_config.save_checkpoint_only_at_epoch_end=true \
++log_config.log_file=$output_dir/train.log \
++metric=acc \
++ckpt_path=$projector_ckpt_path/model.pt \
++train_config.use_peft=true \
++train_config.peft_config.peft_method=$PROMPT_METHOD \
"

cmd="sbatch --account $SLURM_ACCOUNT --job-name slam-${PROMPT} --partition=$SLURM_PARTITION --gpus=${SLURM_GPU_TYPE}:${SLURM_NUM_GPUS} --ntasks=1 --nodes=1"
cmd="${cmd} --cpus-per-task=20 --time=$SLURM_TIME_TRAIN --output=$output_dir/train.%j.out --error=$output_dir/train.%j.err"
# Add dependency if specified
if [ -n "$DEPENDENCY_JOB_ID" ]; then
    cmd="${cmd} --dependency=afterok:$DEPENDENCY_JOB_ID"
fi

job_output=$($cmd --wrap="torchrun \
   --nnodes 1 \
   --nproc_per_node 1 \
   --master_port=23506 \
   $RUN_DIR/finetune_asr.py \
   --config-path "conf" \
   --config-name "${PROMPT}_pp.yaml" \
   ++train_config.enable_fsdp=false \
   ++train_config.enable_ddp=true \
   ++train_config.use_bf16=true \
   ${hydra_args}")

# Extract and return job ID if requested
if [ "$RETURN_JOB_ID" = "true" ]; then
    echo "$job_output" | grep -oP 'Submitted batch job \K\d+'
else
    echo "$job_output"
fi
