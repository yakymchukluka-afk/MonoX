#!/bin/bash
# Script to push hf-build branch to Hugging Face Space
# Run this script to complete the HF deployment

echo "🚀 Pushing MonoX hf-build branch to Hugging Face Space..."

# Set the remote (SSH method - preferred if key is configured)
git remote set-url hf git@hf.co:spaces/lukua/monox

echo "Attempting SSH push..."
git push hf hf-build:main

if [ $? -ne 0 ]; then
    echo "SSH push failed, trying HTTPS..."
    git remote set-url hf https://huggingface.co/spaces/lukua/monox
    git push hf hf-build:main
fi

echo "✅ Push completed! Check your HF Space at: https://huggingface.co/spaces/lukua/monox"