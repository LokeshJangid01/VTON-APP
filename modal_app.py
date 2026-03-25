from pathlib import Path

import modal

APP_NAME = "vton-fastapi"
MODEL_VOLUME_NAME = "vton-model-checkpoints"
MODEL_MOUNT_PATH = "/vol/models"
PROJECT_ROOT_REMOTE = "/root/project"

app = modal.App(APP_NAME)

local_project_root = Path(__file__).resolve().parent


def _ignore_project_file(path: Path) -> bool:
    blocked_parts = {".git", ".conda", ".pip-cache", "checkpoint", "output", "__pycache__"}
    return any(part in blocked_parts for part in path.parts)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        str(local_project_root),
        remote_path=PROJECT_ROOT_REMOTE,
        copy=True,
        ignore=_ignore_project_file,
    )
)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    volumes={MODEL_MOUNT_PATH: model_volume},
    timeout=60 * 60,
)
def sync_checkpoint_from_hf(repo_id: str, revision: str = "main") -> str:
    """
    Download checkpoint files from Hugging Face to a persistent Modal Volume.

    Example:
      modal run modal_app.py::download --repo-id your-org/your-model-repo
    """
    from huggingface_hub import snapshot_download

    target_dir = Path(MODEL_MOUNT_PATH) / "checkpoint" / "checkpoint"
    target_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    model_volume.commit()
    return str(target_dir)


@app.function(
    image=image,
    volumes={MODEL_MOUNT_PATH: model_volume},
    timeout=10 * 60,
)
def inspect_checkpoint() -> str:
    """Return a short listing to verify checkpoint files in the shared volume."""
    model_volume.reload()
    root = Path(MODEL_MOUNT_PATH) / "checkpoint" / "checkpoint"
    required = [
        root / "image_encoder" / "config.json",
        root / "vae" / "config.json",
        root / "denoiser" / "config.json",
    ]
    lines = [f"root_exists={root.exists()}", f"root={root}"]
    for p in required:
        lines.append(f"{p}: {'OK' if p.exists() else 'MISSING'}")
    return "\n".join(lines)


@app.local_entrypoint()
def download(repo_id: str, revision: str = "main"):
    out = sync_checkpoint_from_hf.remote(repo_id=repo_id, revision=revision)
    print(f"Checkpoint synced to: {out}")


@app.function(
    image=image,
    volumes={MODEL_MOUNT_PATH: model_volume},
    timeout=60 * 60,
    gpu="T4",
)
@modal.asgi_app()
def fastapi_app():
    # These env vars let mobile_vton_fastapi/app.py use Modal paths.
    import os
    import sys

    # Ensure this container sees the latest committed files from the shared volume.
    try:
        model_volume.reload()
    except Exception:
        # Fallback paths below still allow startup if volume reload is unavailable.
        pass

    checkpoint_candidates = [
        Path(f"{MODEL_MOUNT_PATH}/checkpoint/checkpoint"),
    ]
    resolved_checkpoint = None
    for candidate in checkpoint_candidates:
        if (candidate / "image_encoder" / "config.json").exists():
            resolved_checkpoint = candidate
            break
    if resolved_checkpoint is None:
        raise RuntimeError(
            "No valid checkpoint found. Expected image_encoder/config.json under one of: "
            + ", ".join(str(p) for p in checkpoint_candidates)
        )

    os.environ.setdefault("CHECKPOINT_DIR", str(resolved_checkpoint))
    os.environ.setdefault("RUNTIME_DIR", "/tmp/mobile_vton_runtime")
    os.environ.setdefault("MODEL_DIR", f"{PROJECT_ROOT_REMOTE}/2026_CVPR_Mobile-VTON")
    os.environ.setdefault("EXTERNAL_TEST_DIR", f"{PROJECT_ROOT_REMOTE}/test/test")
    sys.path.insert(0, PROJECT_ROOT_REMOTE)

    from mobile_vton_fastapi.app import app as web_app

    return web_app
