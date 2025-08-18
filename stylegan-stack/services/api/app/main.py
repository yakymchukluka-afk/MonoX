import os
import sys
import shutil
import uuid
import subprocess
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[3]
REPOS_DIR = ROOT_DIR / "repos"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class GenerateRequest(BaseModel):
    model_path: str = Field(..., description="Absolute or repo-relative path to the .pkl network file")
    seed: int = Field(42, description="Random seed for generation")
    truncation_psi: float = Field(1.0, ge=0.0, le=2.0, description="Truncation psi")
    network: str = Field("stylegan3", regex="^(stylegan3|stylegan2)$", description="Which implementation to use")
    format: str = Field("png", regex="^(png|jpg|jpeg)$", description="Output image format")


app = FastAPI(title="StyleGAN Inference API", version="0.1.0")


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


def _resolve_model_path(model_path: str) -> Path:
    candidate = Path(model_path)
    if candidate.is_file():
        return candidate
    repo_rel = (ROOT_DIR / model_path).resolve()
    if repo_rel.is_file():
        return repo_rel
    raise FileNotFoundError(f"Model file not found: {model_path}")


def _run_generation(req: GenerateRequest) -> Path:
    session_dir = OUTPUT_DIR / f"session_{uuid.uuid4().hex}"
    session_dir.mkdir(parents=True, exist_ok=True)

    model_file = _resolve_model_path(req.model_path)
    img_ext = "jpg" if req.format == "jpeg" else req.format

    if req.network == "stylegan3":
        script_path = REPOS_DIR / "stylegan3" / "gen_images.py"
        if not script_path.is_file():
            raise FileNotFoundError("stylegan3 repo not found or incomplete. Expected gen_images.py")
        cmd = [
            sys.executable,
            str(script_path),
            f"--outdir={session_dir}",
            f"--seeds={req.seed}",
            f"--trunc={req.truncation_psi}",
            f"--network={model_file}",
        ]
    else:
        # stylegan2-ada-pytorch
        script_path = REPOS_DIR / "stylegan2-ada-pytorch" / "generate.py"
        alt_script_path = REPOS_DIR / "stylegan2-ada-pytorch" / "gen_images.py"
        chosen = script_path if script_path.is_file() else alt_script_path
        if not chosen.is_file():
            raise FileNotFoundError("stylegan2-ada-pytorch repo not found or missing generate script")
        cmd = [
            sys.executable,
            str(chosen),
            f"--outdir={session_dir}",
            f"--seeds={req.seed}",
            f"--trunc={req.truncation_psi}",
            f"--network={model_file}",
        ]

    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT_DIR))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Generation failed: {e}")

    # Find the generated image
    images = list(session_dir.glob(f"*.{img_ext}"))
    if not images:
        # Try png as a fallback if JPEG selected
        if img_ext != "png":
            images = list(session_dir.glob("*.png"))
    if not images:
        raise RuntimeError("No images were generated")
    return images[0]


@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        image_path = _run_generation(req)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return FileResponse(
        path=str(image_path),
        media_type=f"image/{req.format if req.format != 'jpeg' else 'jpg'}",
        filename=image_path.name,
    )

