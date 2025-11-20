#!/bin/bash
# Determine script and run directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(dirname "$SCRIPT_DIR")"

# Source environment variables
source "$SCRIPT_DIR/config_defaults.sh"

cd $RUN_DIR

# Script-specific configuration
output_dir=$RUN_DIR/exp/$EXPERIMENT_NAME/base/${PROMPT}/${BASE_CKPT_FOLDER}
output_file=$output_dir/decode_output

cmd="sbatch --account $SLURM_ACCOUNT --job-name dec-slam-${PROMPT} --partition=$SLURM_PARTITION --gpus=${SLURM_GPU_TYPE}:${SLURM_NUM_GPUS} --ntasks=1 --nodes=1"
cmd="${cmd} --cpus-per-task=10 --time=$SLURM_TIME_DECODE --output=$output_dir/decode.%j.out --error=$output_dir/decode.%j.err"
# Add dependency if specified
if [ -n "$DEPENDENCY_JOB_ID" ]; then
    cmd="${cmd} --dependency=afterok:$DEPENDENCY_JOB_ID"
fi

job_output=$($cmd --wrap="python $RUN_DIR/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "${PROMPT}.yaml" \
        hydra.run.dir=$output_dir \
        ++model_config.llm_name="vicuna-1b" \
        ++model_config.llm_path=$LLM_PATH \
        ++model_config.llm_dim=$LLM_DIM \
        ++model_config.encoder_name=wavlm \
        ++model_config.normalize=true \
        ++dataset_config.normalize=true \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$SPEECH_ENCODER_PATH \
        ++model_config.encoder_dim=$SPEECH_ENCODER_DIM \
        ++model_config.encoder_projector=linear \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$TEST_DATA_PATH \
        ++dataset_config.input_type=raw \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=4 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$output_file \
        ++log_config.log_file=$output_dir/decode.log \
        ++ckpt_path=$output_dir/model.pt \
        ++train_config.use_bf16=true")

# Extract and return job ID if requested
if [ "$RETURN_JOB_ID" = "true" ]; then
    echo "$job_output" | grep -oP 'Submitted batch job \K\d+'
else
    echo "$job_output"
fi
