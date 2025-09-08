# Legacy Branch Archives

This directory contains archived snapshots of experimental branches that were consolidated during the repository cleanup on 2025-09-08.

## Archived Branches

### runpod-hf-build
- **Backup Tag**: `backup/runpod-hf-build-20250908`
- **Description**: RunPod setup with HuggingFace integration
- **Contents**: RunPod container setup scripts with HF authentication

### feat-sg2-runpod-setup
- **Backup Tag**: `backup/feat-sg2-runpod-setup-20250908`
- **Description**: Feature branch for StyleGAN2 RunPod setup
- **Contents**: Initial RunPod configuration and training scripts

### cursor-setup-and-run-monox-training-on-runpod-97f8
- **Backup Tag**: `backup/cursor-setup-and-run-monox-training-on-runpod-97f8-20250908`
- **Description**: Cursor AI assisted RunPod training setup
- **Contents**: Automated RunPod training configuration and scripts

### runpod-training
- **Backup Tag**: `backup/runpod-training-20250908`
- **Description**: General RunPod training experiments
- **Contents**: Various RunPod training configurations and scripts

### hf-training
- **Backup Tag**: `backup/hf-training-20250908`
- **Description**: HuggingFace training experiments
- **Contents**: HF-specific training scripts and configurations

### collab-stylegen-training
- **Backup Tag**: `backup/collab-stylegen-training-20250908`
- **Description**: Google Colab StyleGAN training experiments
- **Contents**: Colab notebooks and training scripts

## Recovery

To recover any of these branches:

```bash
# Create a new branch from the backup tag
git checkout -b <new-branch-name> backup/<backup-tag-name>

# Or restore the exact branch name
git checkout -b <original-branch-name> backup/<backup-tag-name>
```

## Note

These archives are preserved for historical reference. The active development has been consolidated into:
- `runpod/sg2-1024` - Clean RunPod training with official StyleGAN2-ADA
- `archive/huggingface-training-20250908` - HuggingFace training archive
- `archive/colab-training-20250908` - Google Colab training archive