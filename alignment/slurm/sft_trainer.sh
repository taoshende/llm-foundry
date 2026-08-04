#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn about SLURM sbatch options at:
# - https://slurm.schedmd.com/sbatch.html
#
# Learn about job submissions (Marvin|Bender) at:
# - https://wiki.hpc.uni-bonn.de/en/running_jobs
#
# Learn about Marvin|Bender dual software stacks at:
# - https://wiki.hpc.uni-bonn.de/en/dualstacks
#############################################
#SBATCH --partition=A40devel             # <-- Change to your partition
#SBATCH --job-name=sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --gpus=1

#############################################
# Working Directory Setup
#############################################

# Set this to your workspace root (where you have the .venv and .modules.sh files).
workdir="/home/s6shtaoo/CAISA"
mkdir -p "$workdir/alignment/logs"
cd "$workdir"
ulimit -c 0

out="$workdir/alignment/logs/out-sft-trainer.$SLURM_JOB_ID"
err="$workdir/alignment/logs/err-sft-trainer.$SLURM_JOB_ID"

#############################################
# Modules & Libraries Setup
#############################################

source $workdir/.modules.sh > "$out" 2>&1
# python3 -m venv $workdir/.venv_trl
source "$workdir/.venv_trl/bin/activate"

# ===== Upgrade PIP =====
# pip3 install --upgrade pip

# ===== LLM Foundry Install (for Bender) =====

# pip3 install \
#     setuptools \
#     liger-kernel==0.8.0 \
#     kernels==0.13.0 \
#     trl \
#     vllm \
#     codecarbon==3.0.6 \
#     wandb==0.27.2 \
#     --no-cache-dir

# pip3 install wheel==0.45.1 packaging==25.0 --no-cache-dir
# pip3 install \
#     torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
#     --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir

# FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip3 install \
#     https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu124torch2.6-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl \
#     --no-cache-dir

#############################################
# Environment Setup
#############################################
# PyTorch NCCL environment variables:
# - https://github.com/pytorch/pytorch/blob/main/docs/source/cuda_environment_variables.rst
#
# PyTorch Distributed Documentation:
# - https://github.com/pytorch/pytorch/blob/main/docs/source/distributed.md
#
# NCCL Documentation:
# - https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
#############################################

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.tmp/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export HF_TOKEN=""
export WANDB_TOKEN=""
export WANDB_DIR="$HF_DATASETS_CACHE/wandb"
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache/$SLURM_JOB_ID"
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_IB_TIMEOUT=24
export NCCL_IB_RETRY_CNT=7
export TORCH_FR_BUFFER_SIZE=1000
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
# export NCCL_DEBUG=INFO # Uncomment for NCCL debugging
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=$SLURM_NTASKS_PER_NODE
export NUM_PROCESSES=$SLURM_NTASKS
export NUM_MACHINES=$SLURM_NNODES
export head_node_ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export CHECKPOINT_DIR="$workdir/alignment/sft-qwen-base/checkpoints/checkpoints-$SLURM_JOB_ID"
export CLEAN_CACHE="1"  # <-- Set to "1" to clean cache after job completion

mkdir -p $CHECKPOINT_DIR

echo "# [${SLURM_JOB_ID}] Job started on $SLURM_JOB_NODELIST at: $(date)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# [${SLURM_JOB_ID}] GLIBC version: $(ldd --version | head -n1)" >> "$out"
echo "# [${SLURM_JOB_ID}] Working directory: $workdir" >> "$out"
echo "# [${SLURM_JOB_ID}] Python executable: $(which python3) — $(python3 --version)" >> "$out"

#############################################
# Main Job Execution (Distributed Training)
#############################################
# Accelerate Documentation
# - https://huggingface.co/docs/accelerate/package_reference/cli
#############################################

export LAUNCHER="accelerate launch --config_file $workdir/llm-foundry/alignment/.ddp_config.yaml"

export PYTHON_FILE="$workdir/llm-foundry/alignment/sft_trainer.py"

export ARGS="--dataset_type parquet \
--train_dataset_dir $workdir/data/capybara/train-00000-of-00001.parquet \
--shuffle_dataset \
--cache_dir $HF_DATASETS_CACHE \
--num_proc 8 \
--model_name_or_path $workdir/alignment/sft-qwen-base/model \
--chat_template_path $workdir/alignment/sft-qwen-base/chat_template.jinja \
--checkpoint_dir $CHECKPOINT_DIR \
--max_length 1024 \
--save_steps 2000 \
--logging_steps 1 \
--packing \
--assistant_only_loss \
--use_liger_kernel \
--learning_rate 0.00005 \
--weight_decay 0.1 \
--lr_scheduler_type cosine \
--warmup_ratio 0.1 \
--num_train_epochs 1 \
--attn_implementation flash_attention_2 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--bf16 \
--tf32 \
--gradient_checkpointing \
"

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $ARGS"
$CMD 1>>"$out" 2>>"$err"

#############################################
# End of Script
#############################################
# Clean HF_DATASETS_CACHE folder if requested
if [ "$CLEAN_CACHE" = "1" ]; then
    echo "# [${SLURM_JOB_ID}] Cleaning HF_DATASETS_CACHE" >> "$out"
    if [ -d "$HF_DATASETS_CACHE" ]; then
        find "$HF_DATASETS_CACHE" -mindepth 1 -delete 2>/dev/null || true
    fi
else
    echo "# [${SLURM_JOB_ID}] Skipping cache cleanup (CLEAN_CACHE=$CLEAN_CACHE)" >> "$out"
fi

echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "$out"
cp "$out" "$CHECKPOINT_DIR/logs.txt"
cp "${BASH_SOURCE[0]}" "$CHECKPOINT_DIR/job.sh"
