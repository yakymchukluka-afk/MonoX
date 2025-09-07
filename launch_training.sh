#!/bin/bash
cd /workspace/code/MonoX/train/runpod-hf
export PYTHONPATH="$PWD/vendor/stylegan2ada:$PWD/vendor/styleganv/src:$PWD:$PYTHONPATH"
python /workspace/runpod_train.py