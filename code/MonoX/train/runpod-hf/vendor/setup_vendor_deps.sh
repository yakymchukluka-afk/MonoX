#!/bin/bash
set -euxo pipefail

cd "$(dirname "$0")"

# Clone NVLabs StyleGAN2-ADA
if [ ! -d "stylegan2ada" ]; then
  git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git stylegan2ada
fi

# Clone StyleGAN-V
if [ ! -d "styleganv" ]; then
  git clone https://github.com/universome/stylegan-v.git styleganv
fi

echo "Vendor dependencies set up successfully!"