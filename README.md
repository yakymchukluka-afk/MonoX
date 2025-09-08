# MonoX - StyleGAN Training Repository

This repository contains multiple training approaches for StyleGAN models, organized into specialized branches for different environments and use cases.

## 🚀 Active Branches

### [runpod/sg2-1024](https://github.com/yakymchukluka-afk/MonoX/tree/runpod/sg2-1024) - **Active RunPod Training**
Clean, production-ready RunPod training setup using official NVLabs StyleGAN2-ADA.

**Features:**
- Official StyleGAN2-ADA submodule integration
- Minimal, idempotent RunPod scripts
- Git LFS support for large datasets
- Comprehensive documentation

**Quick Start:**
```bash
git checkout runpod/sg2-1024
bash scripts/runpod/bootstrap.sh
bash scripts/runpod/make_dataset_zip.sh /path/to/dataset
bash scripts/runpod/train.sh
```

## 📚 Archive Branches

### [archive/huggingface-training-20250908](https://github.com/yakymchukluka-afk/MonoX/tree/archive/huggingface-training-20250908) - **HuggingFace Archive**
Frozen HuggingFace training setup with HF Spaces integration.

**Contents:**
- HF Space configuration and setup scripts
- HuggingFace authentication utilities
- Training configurations for HF environment
- Documentation and troubleshooting guides

### [archive/colab-training-20250908](https://github.com/yakymchukluka-afk/MonoX/tree/archive/colab-training-20250908) - **Google Colab Archive**
Frozen Google Colab training setup with Jupyter notebooks.

**Contents:**
- Colab setup scripts and installation utilities
- Interactive Jupyter notebooks for training
- Colab-specific troubleshooting documentation
- StyleGAN-V integration

### [archive/runpod-experiments-20250908](https://github.com/yakymchukluka-afk/MonoX/tree/archive/runpod-experiments-20250908) - **Experimental RunPod Archive**
Consolidated archive of all experimental RunPod branches.

**Contents:**
- Legacy experimental branches in `legacy/` directories
- Backup tags for all deleted branches
- Historical RunPod configurations and scripts
- Complete experimental work preservation

## 🔧 Recovery

All deleted experimental branches are preserved with backup tags:

```bash
# List all backup tags
git tag -l "backup/*"

# Restore a specific branch
git checkout -b <branch-name> backup/<backup-tag-name>
```

## 📋 Branch Status

| Branch | Status | Purpose | Last Updated |
|--------|--------|---------|--------------|
| `runpod/sg2-1024` | ✅ Active | Production RunPod training | 2025-09-08 |
| `archive/huggingface-training-20250908` | 📦 Frozen | HF training archive | 2025-09-08 |
| `archive/colab-training-20250908` | 📦 Frozen | Colab training archive | 2025-09-08 |
| `archive/runpod-experiments-20250908` | 📦 Frozen | Experimental archive | 2025-09-08 |

## 🛠️ Development

- **Active Development**: Use `runpod/sg2-1024` for new features
- **Historical Reference**: Check archive branches for previous implementations
- **Recovery**: Use backup tags to restore any deleted experimental work

## 📖 Documentation

Each branch contains its own comprehensive documentation:
- `README-RUNPOD.md` - RunPod training guide
- `README-ARCHIVE-HF.md` - HuggingFace archive documentation
- `README-ARCHIVE-COLAB.md` - Colab archive documentation
- `legacy/README.md` - Experimental branches documentation

## 🔒 Safety

- All experimental work is preserved in backup tags
- No history has been rewritten
- All branches can be recovered if needed
- Clean separation between active and archived work

---

**Repository Cleanup Date**: 2025-09-08  
**Total Branches**: 4 (1 active, 3 archives)  
**Backup Tags**: 6 (for all deleted experimental branches)