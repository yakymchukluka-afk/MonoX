# Google Colab Training Archive

This branch contains the frozen Google Colab training setup for MonoX as of 2025-09-08.

## Contents

- **Colab setup scripts**: `colab_install.py`, `colab_install_v2.py` for environment setup
- **Colab notebooks**: `MonoX_GPU_Colab.ipynb` for interactive training
- **Documentation**: 
  - `COLAB_SETUP.md` - Basic setup instructions
  - `COLAB_NUCLEAR_SETUP.md` - Advanced setup for complex cases
  - `COLAB_MODULE_FIX.md` - Module import fixes
  - `COLAB_TROUBLESHOOTING.md` - Common issues and solutions
- **Configuration**: `configs/` directory with training configurations
- **Core training code**: `src/` directory with infrastructure
- **Scripts**: `scripts/` directory with training scripts
- **Samples**: `samples/` directory with example outputs
- **Logs**: `logs/` directory for training logs
- **StyleGAN-V integration**: `.external/stylegan-v/` submodule

## Usage Instructions

1. **Open the Colab notebook**: Use `MonoX_GPU_Colab.ipynb` in Google Colab
2. **Run setup**: Execute `colab_install.py` or `colab_install_v2.py` for dependencies
3. **Configure training**: Modify configs as needed
4. **Start training**: Follow the notebook instructions

## Environment Setup

The Colab environment requires:
- GPU runtime (T4, V100, or A100)
- Custom CUDA kernel compilation for StyleGAN
- Specific package versions for compatibility

## Note

This archive preserves the Google Colab-specific training setup that was working at the time of archiving. All non-Colab related files have been moved to `_not_in_this_archive/` directory.

## Troubleshooting

If you encounter issues, refer to:
- `COLAB_TROUBLESHOOTING.md` for common problems
- `COLAB_MODULE_FIX.md` for import/module issues
- `COLAB_NUCLEAR_SETUP.md` for complex setup scenarios