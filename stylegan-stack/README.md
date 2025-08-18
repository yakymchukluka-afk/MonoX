## StyleGAN Stack

End-to-end scaffolding to train and serve StyleGAN on your art dataset.

### What’s included
- Repos: NVLabs `stylegan3` and `stylegan2-ada-pytorch` under `repos/`
- FastAPI inference service under `services/api`
- Dockerfile and `docker-compose.yml`
- Scripts for dataset prep and training in `scripts/`
- Terraform skeleton for AWS in `infra/terraform`

### Prerequisites
- NVIDIA GPU machine for training/inference (CUDA 11.x recommended) or a GPU cloud instance (AWS g5/g6)
- Docker with NVIDIA runtime for GPU in containers (optional but recommended)
- Python 3.10+ if running locally without Docker

### Dataset preparation
Your images should be in a single directory. Prepare a StyleGAN dataset zip:

```bash
./scripts/prepare_dataset.sh /path/to/images /workspace/stylegan-stack/data/artpieces.zip
```

Notes:
- Images should be square or will be center-cropped by dataset tool
- Prefer at least 5k–20k images; higher counts improve diversity

### Training (StyleGAN3)
Example single-GPU training run:

```bash
./scripts/train_stylegan3.sh \
  /workspace/stylegan-stack/data/artpieces.zip \
  /workspace/stylegan-stack/models/exp1 \
  stylegan3-t \
  1 \
  512 \
  32
```

Arguments: `<dataset_zip> <outdir> <cfg: stylegan3-t|stylegan3-r> <gpus> <resolution> <batch>`

### Inference (StyleGAN3)
Generate images from a trained `.pkl`:

```bash
./scripts/infer_stylegan3.sh /workspace/stylegan-stack/models/exp1/network-snapshot-000xxx.pkl 42 /workspace/stylegan-stack/output
```

### API (local)
Run with Docker (recommended):

```bash
docker compose up --build
# Health check
curl http://localhost:8000/health
# Generate (supports stylegan3 or stylegan2)
curl -X POST http://localhost:8000/generate -H 'Content-Type: application/json' \
  -d '{
        "model_path":"/app/models/network.pkl",
        "seed":42,
        "truncation_psi":1.0,
        "network":"stylegan3",
        "format":"png"
      }' --output sample.png
```

### Cloud (AWS Terraform skeleton)
The Terraform in `infra/terraform` sets up:
- S3 buckets for datasets and outputs
- ECR repo for API image
- ECS cluster/service scaffold for GPU tasks

Initialize and plan:

```bash
cd infra/terraform
terraform init
terraform plan -var="project=stylegan-stack" -var="aws_region=us-east-1"
```

### Notes
- There is no official "StyleGAN5" from NVLabs at this time. This stack uses StyleGAN3 (and StyleGAN2-ADA) which are state-of-the-art for GAN-based image synthesis.
- For training at high resolutions (512–1024), prefer 24–48GB VRAM GPUs.

