#!/bin/bash
# Demo script showing the complete StyleGAN2-ADA training setup

echo "🎯 MonoX StyleGAN2-ADA Training Setup Demo"
echo "=========================================="
echo ""

echo "📋 This demo shows the complete training workflow:"
echo "1. Repository setup and submodule initialization"
echo "2. System dependencies installation"
echo "3. Dataset preparation"
echo "4. Training execution"
echo "5. Monitoring and management"
echo ""

echo "🔧 Current Setup Status:"
echo "========================"

# Check if we're in the right branch
echo -n "Branch: "
git branch --show-current

# Check if submodule is initialized
if [ -d "vendor/stylegan2ada" ] && [ -f "vendor/stylegan2ada/train.py" ]; then
    echo "✅ StyleGAN2-ADA submodule: Initialized"
else
    echo "❌ StyleGAN2-ADA submodule: Not found"
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✅ Python virtual environment: Created"
else
    echo "❌ Python virtual environment: Not found"
fi

# Check if scripts are executable
if [ -x "scripts/runpod/bootstrap.sh" ]; then
    echo "✅ RunPod scripts: Ready"
else
    echo "❌ RunPod scripts: Not executable"
fi

echo ""

echo "📚 Available Scripts:"
echo "===================="
echo "• bootstrap.sh     - System setup and dependencies"
echo "• make_dataset_zip.sh - Dataset preparation"
echo "• train.sh         - Training execution"
echo "• monitor.sh       - Training monitoring"
echo ""

echo "🚀 Quick Start Commands:"
echo "======================="
echo ""
echo "1. System Setup (if not done):"
echo "   bash scripts/runpod/bootstrap.sh"
echo ""
echo "2. Prepare Dataset:"
echo "   bash scripts/runpod/make_dataset_zip.sh /path/to/images /workspace/datasets/mydataset.zip 1024 1024"
echo ""
echo "3. Start Training:"
echo "   bash scripts/runpod/train.sh /workspace/out/sg2 /workspace/datasets/mydataset.zip 8"
echo ""
echo "4. Monitor Training:"
echo "   bash scripts/runpod/monitor.sh"
echo "   tmux attach -t monox"
echo ""

echo "📖 Documentation:"
echo "================"
echo "• README-RUNPOD.md - Complete training guide"
echo "• configs/runpod/sg2-1024.example.yaml - Configuration examples"
echo "• vendor/stylegan2ada/README.md - Official StyleGAN2-ADA docs"
echo ""

echo "🎯 Training Configurations:"
echo "=========================="
echo "• auto      - Automatic selection (recommended for new datasets)"
echo "• stylegan2 - StyleGAN2 config F (high-quality results)"
echo "• paper1024 - Paper configuration for 1024x1024"
echo "• paper512  - Paper configuration for 512x512"
echo ""

echo "📊 Expected Training Times (A100 8x GPU):"
echo "========================================="
echo "• 512x512   - 2h 48m (1000 kimg) / 2d 22h (25000 kimg)"
echo "• 1024x1024 - 5h 54m (1000 kimg) / 6d 03h (25000 kimg)"
echo ""

echo "🔍 Troubleshooting:"
echo "=================="
echo "• Check GPU: nvidia-smi"
echo "• Check logs: tail -f /workspace/out/sg2/log.txt"
echo "• Check tmux: tmux list-sessions"
echo "• Restart training: tmux kill-session -t monox && bash scripts/runpod/train.sh ..."
echo ""

echo "✅ Setup is ready for RunPod A100 training!"
echo "   For questions, see README-RUNPOD.md"