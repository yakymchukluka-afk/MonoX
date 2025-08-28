#!/bin/bash
# StyleGAN-V Training Script for Mono Project
# This script sets up and launches StyleGAN-V training with the specified configuration

set -e  # Exit on any error

# Configuration
STYLEGAN_DIR="/workspace/stylegan-stack/stylegan-v"
CONFIG_FILE="/workspace/stylegan-stack/training_config.yaml"
DATASET_PATH="/workspace/stylegan-stack/data/originals/"
CHECKPOINT_DIR="/workspace/stylegan-stack/models/checkpoints/"
LOGS_DIR="/workspace/stylegan-stack/logs/"
PREVIEWS_DIR="/workspace/stylegan-stack/generated_previews/"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --resume CHECKPOINT    Resume training from checkpoint"
    echo "  --gpus N               Number of GPUs to use (default: auto-detect)"
    echo "  --batch-size N         Batch size (default: 16)"
    echo "  --dry-run              Print configuration and exit"
    echo "  --help                 Display this help message"
    exit 1
}

# Default values
RESUME_CHECKPOINT=""
NUM_GPUS="auto"
BATCH_SIZE="16"
DRY_RUN="false"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Setup environment
echo "Setting up environment..."
export PATH="/home/ubuntu/.local/bin:$PATH"
cd "$STYLEGAN_DIR"

# Auto-detect GPUs if not specified
if [ "$NUM_GPUS" = "auto" ]; then
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 1)")
    echo "Auto-detected $NUM_GPUS GPU(s)"
fi

# Verify dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Warning: Dataset directory $DATASET_PATH does not exist yet."
    echo "Please upload your dataset before starting training."
fi

# Create output directories
mkdir -p "$CHECKPOINT_DIR" "$LOGS_DIR" "$PREVIEWS_DIR"

# Build training command
TRAIN_CMD="python3 src/infra/launch.py"
TRAIN_CMD+=" hydra.run.dir=."
TRAIN_CMD+=" hydra.output_subdir=null"
TRAIN_CMD+=" hydra/job_logging=disabled" 
TRAIN_CMD+=" hydra/hydra_logging=disabled"
TRAIN_CMD+=" +exp_suffix=mono_project_1024px"
TRAIN_CMD+=" dataset=mono_project"
TRAIN_CMD+=" training=mono_training"
TRAIN_CMD+=" num_gpus=$NUM_GPUS"
TRAIN_CMD+=" training.batch_size=$BATCH_SIZE"
TRAIN_CMD+=" training.dry_run=$DRY_RUN"

# Add resume checkpoint if specified
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_CMD+=" training.resume=$RESUME_CHECKPOINT"
    echo "Resuming training from: $RESUME_CHECKPOINT"
fi

# Display configuration
echo "============================================"
echo "StyleGAN-V Training Configuration"
echo "============================================"
echo "Dataset path: $DATASET_PATH"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Logs dir: $LOGS_DIR"
echo "Previews dir: $PREVIEWS_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch size: $BATCH_SIZE"
echo "Dry run: $DRY_RUN"
echo "Resume checkpoint: ${RESUME_CHECKPOINT:-None}"
echo "============================================"

# Check PyTorch and CUDA
echo "Checking PyTorch installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('Warning: CUDA not available, will use CPU (very slow)')
"

if [ "$DRY_RUN" = "true" ]; then
    echo "Dry run mode - would execute:"
    echo "$TRAIN_CMD"
    exit 0
fi

# Launch training
echo "Starting training..."
echo "Command: $TRAIN_CMD"
echo "============================================"

# Run with logging
exec $TRAIN_CMD 2>&1 | tee "$LOGS_DIR/training_$(date +%Y%m%d_%H%M%S).log"