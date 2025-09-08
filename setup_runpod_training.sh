#!/bin/bash
set -e

echo "🎯 MonoX StyleGAN2-ADA RunPod Training Setup"
echo "============================================="
echo ""

# Check if we're in the right directory
if [ ! -f "vendor/stylegan2ada/train.py" ]; then
    echo "❌ Error: StyleGAN2-ADA submodule not found"
    echo "   Please run: git submodule update --init --recursive"
    exit 1
fi

echo "✅ StyleGAN2-ADA submodule found"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /workspace/out/sg2
mkdir -p /workspace/datasets
mkdir -p /workspace/logs

echo "✅ Directories created"
echo ""

# Check GPU availability
echo "🔍 Checking GPU availability..."
source /workspace/venv/bin/activate
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.device_count()} GPUs')
    for i in range(torch.cuda.device_count()):
        print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'   Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('❌ CUDA not available')
    print('   Please run: bash scripts/runpod/bootstrap.sh')
    exit(1)
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "=============="
echo ""
echo "1. Prepare your dataset:"
echo "   bash scripts/runpod/make_dataset_zip.sh /path/to/dataset /workspace/datasets/mydataset.zip 1024 1024"
echo ""
echo "2. Start training:"
echo "   bash scripts/runpod/train.sh /workspace/out/sg2 /workspace/datasets/mydataset.zip 8"
echo ""
echo "3. Monitor training:"
echo "   bash scripts/runpod/monitor.sh"
echo "   tmux attach -t monox"
echo ""
echo "📚 Documentation:"
echo "   README-RUNPOD.md - Complete training guide"
echo "   vendor/stylegan2ada/README.md - Official StyleGAN2-ADA docs"
echo ""
echo "🚀 Ready to start training!"