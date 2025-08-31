#!/bin/bash
# Pre-build script to handle git configuration before HF Spaces infrastructure
# This runs before HF tries to configure git, preventing permission errors

set -e

echo "🔧 Pre-configuring git to prevent permission errors..."

# Create home directory if it doesn't exist
mkdir -p $HOME

# Pre-configure git with the expected values
git config --global user.name "lukua"
git config --global user.email "lukua@users.noreply.huggingface.co"

# Set safe permissions
chmod 644 $HOME/.gitconfig 2>/dev/null || true

echo "✅ Git pre-configuration complete"