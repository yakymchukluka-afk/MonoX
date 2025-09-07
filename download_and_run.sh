#!/bin/bash
# Download and run simple MonoX training script

cd /workspace/code/MonoX/train/runpod-hf

# Download the simple training script
curl -o monox/simple_train.py https://raw.githubusercontent.com/yakymchukluka-afk/MonoX/feat/styleganv-loss-default/simple_monox_train.py

# Make it executable
chmod +x monox/simple_train.py

# Run the training
export PYTHONPATH="$PWD/vendor/stylegan2ada:$PWD/vendor/styleganv/src:$PWD:$PYTHONPATH"
python monox/simple_train.py