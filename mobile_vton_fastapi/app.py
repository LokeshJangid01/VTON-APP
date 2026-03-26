import asyncio
import base64
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(APP_DIR.parent / "2026_CVPR_Mobile-VTON")))
DATASET_DIR = Path(os.getenv("DATASET_DIR", str(MODEL_DIR / "Mobile_VTON" / "VITON-HD")))
EXTERNAL_TEST_DIR = Path(os.getenv("EXTERNAL_TEST_DIR", str(APP_DIR.parent / "test" / "test")))
SOURCE_CLOTH_DIR = EXTERNAL_TEST_DIR / "cloth"
SOURCE_DESCRIPTIONS_CANDIDATES = [
    EXTERNAL_TEST_DIR / "image_descriptions.txt",
    DATASET_DIR / "test" / "image_descriptions.txt",
]
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", str(MODEL_DIR / "checkpoint" / "checkpoint")))
RUNTIME_DIR = Path(os.getenv("RUNTIME_DIR", str(APP_DIR / "runtime")))
JOBS_DIR = RUNTIME_DIR / "jobs"
RUNTIME_VOLUME_NAME = os.getenv("RUNTIME_VOLUME_NAME", "")

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
inference_lock = asyncio.Lock()

app = FastAPI(title="Mobile-VTON FastAPI")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


def _load_descriptions() -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for desc_file in SOURCE_DESCRIPTIONS_CANDIDATES:
        if not desc_file.exists():
            continue
        for line in desc_file.read_text(encoding="utf-8").splitlines():
            if ": " in line:
                name, text = line.split(": ", 1)
                descriptions[name.strip()] = text.strip()
        if descriptions:
            break
    return descriptions


def _safe_name(filename: str) -> str:
    return Path(filename).name.replace(" ", "_")


def _sync_runtime_volume(commit: bool) -> None:
    if not RUNTIME_VOLUME_NAME:
        return
    try:
        import modal

        vol = modal.Volume.from_name(RUNTIME_VOLUME_NAME, create_if_missing=False)
        if commit:
            vol.commit()
        else:
            vol.reload()
    except Exception:
        # Best-effort only: app still works on single-container routing.
        pass


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_file = APP_DIR / "static" / "index.html"
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/api/cloths")
async def list_cloths():
    if not SOURCE_CLOTH_DIR.exists():
        raise HTTPException(status_code=500, detail=f"Cloth dir not found: {SOURCE_CLOTH_DIR}")
    cloths = sorted(
        [p.name for p in SOURCE_CLOTH_DIR.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_SUFFIXES]
    )
    return {"cloths": cloths}


@app.get("/api/cloths/{cloth_name}")
async def get_cloth_image(cloth_name: str):
    safe = _safe_name(cloth_name)
    path = SOURCE_CLOTH_DIR / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="Cloth not found")
    return FileResponse(path)


@app.get("/api/results/{job_id}/{filename}")
async def get_result(job_id: str, filename: str):
    safe = _safe_name(filename)
    path = JOBS_DIR / job_id / "output" / safe
    if not path.exists():
        _sync_runtime_volume(commit=False)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(path)


def _prepare_dataset_data(
    job_id: str,
    person_image: UploadFile,
    cloth_name: str,
    descriptions: dict[str, str],
) -> tuple[str, str]:
    test_dir = DATASET_DIR / "test"
    image_dir = test_dir / "image"
    cloth_dir = test_dir / "cloth"
    image_dir.mkdir(parents=True, exist_ok=True)
    cloth_dir.mkdir(parents=True, exist_ok=True)

    cloth_src = SOURCE_CLOTH_DIR / cloth_name
    cloth_dst = cloth_dir / cloth_name
    if not cloth_src.exists():
        raise HTTPException(status_code=400, detail=f"Cloth '{cloth_name}' not found in test/cloth")
    shutil.copy2(cloth_src, cloth_dst)

    incoming_name = _safe_name(person_image.filename or "person.jpg")
    ext = Path(incoming_name).suffix.lower()
    if ext not in ALLOWED_IMAGE_SUFFIXES:
        ext = ".jpg"
    person_filename = f"person_api_{job_id}{ext}"
    person_dst = image_dir / person_filename
    with person_dst.open("wb") as f:
        shutil.copyfileobj(person_image.file, f)

    pair_file = DATASET_DIR / "test_pairs.txt"
    pair_file.write_text(f"{person_filename} {cloth_name}\n", encoding="utf-8")

    cloth_desc = descriptions.get(cloth_name, "shirt")
    desc_file = test_dir / "image_descriptions.txt"
    existing_lines = []
    existing_map = {}
    if desc_file.exists():
        existing_lines = desc_file.read_text(encoding="utf-8").splitlines()
        for line in existing_lines:
            if ": " in line:
                name, text = line.split(": ", 1)
                existing_map[name.strip()] = text.strip()
    if cloth_name not in existing_map:
        existing_lines.append(f"{cloth_name}: {cloth_desc}")
        desc_file.write_text("\n".join(existing_lines).strip() + "\n", encoding="utf-8")
    return person_filename, Path(person_filename).stem


def _prepare_dataset_data_from_bytes(
    job_id: str,
    person_image_bytes: bytes,
    person_image_filename: str,
    cloth_name: str,
    descriptions: dict[str, str],
) -> tuple[str, str]:
    test_dir = DATASET_DIR / "test"
    image_dir = test_dir / "image"
    cloth_dir = test_dir / "cloth"
    image_dir.mkdir(parents=True, exist_ok=True)
    cloth_dir.mkdir(parents=True, exist_ok=True)

    cloth_src = SOURCE_CLOTH_DIR / cloth_name
    cloth_dst = cloth_dir / cloth_name
    if not cloth_src.exists():
        raise HTTPException(status_code=400, detail=f"Cloth '{cloth_name}' not found in test/cloth")
    shutil.copy2(cloth_src, cloth_dst)

    incoming_name = _safe_name(person_image_filename or "person.jpg")
    ext = Path(incoming_name).suffix.lower()
    if ext not in ALLOWED_IMAGE_SUFFIXES:
        ext = ".jpg"
    person_filename = f"person_api_{job_id}{ext}"
    person_dst = image_dir / person_filename
    person_dst.write_bytes(person_image_bytes)

    pair_file = DATASET_DIR / "test_pairs.txt"
    pair_file.write_text(f"{person_filename} {cloth_name}\n", encoding="utf-8")

    cloth_desc = descriptions.get(cloth_name, "shirt")
    desc_file = test_dir / "image_descriptions.txt"
    existing_lines = []
    existing_map = {}
    if desc_file.exists():
        existing_lines = desc_file.read_text(encoding="utf-8").splitlines()
        for line in existing_lines:
            if ": " in line:
                name, text = line.split(": ", 1)
                existing_map[name.strip()] = text.strip()
    if cloth_name not in existing_map:
        existing_lines.append(f"{cloth_name}: {cloth_desc}")
        desc_file.write_text("\n".join(existing_lines).strip() + "\n", encoding="utf-8")
    return person_filename, Path(person_filename).stem


def _decode_image_b64(image_b64: str) -> bytes:
    data = image_b64.strip()
    if "," in data and data.lower().startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        return base64.b64decode(data, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid person_image_b64") from exc


def _run_inference(job_dir: Path, person_filename: str, steps: int, guidance_scale: float) -> tuple[str, str]:
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "inference.py",
        "--data_dir",
        str(DATASET_DIR),
        "--output_dir",
        str(output_dir),
        "--order",
        "unpaired",
        "--height",
        "1024",
        "--width",
        "768",
        "--test_batch_size",
        "1",
        "--num_workers",
        "0",
        "--num_inference_steps",
        str(steps),
        "--guidance_scale",
        str(guidance_scale),
        "--mixed_precision",
        "no",
        "--person_image_name",
        person_filename,
        "--checkpoint_path",
        str(CHECKPOINT_DIR),
    ]

    result = subprocess.run(
        cmd,
        cwd=MODEL_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    logs = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    if result.returncode != 0:
        raise RuntimeError(logs.strip() or f"inference.py failed with code {result.returncode}")
    return str(output_dir), logs


def _build_tryon_response(
    job_id: str,
    output_dir: str,
    person_stem: str,
    cloth_name: str,
    logs: str,
):
    candidates = sorted(Path(output_dir).glob(f"{person_stem}_{cloth_name}*"))
    if not candidates:
        candidates = sorted(Path(output_dir).glob("*"))
    if not candidates:
        raise HTTPException(status_code=500, detail="Inference finished but no output image was generated")

    _sync_runtime_volume(commit=True)
    result_file = candidates[0].name
    return {
        "job_id": job_id,
        "result_url": f"/api/results/{job_id}/{result_file}",
        "result_file": result_file,
        "logs_tail": "\n".join(logs.strip().splitlines()[-25:]),
    }


class TryOnJsonRequest(BaseModel):
    person_image_b64: str
    person_image_filename: str = "person.jpg"
    cloth_name: str
    num_inference_steps: int = 8
    guidance_scale: float = 2.0


@app.post("/api/tryon")
async def run_tryon(
    person_image: UploadFile = File(...),
    cloth_name: str = Form(...),
    num_inference_steps: int = Form(8),
    guidance_scale: float = Form(2.0),
):
    if num_inference_steps < 1 or num_inference_steps > 50:
        raise HTTPException(status_code=400, detail="num_inference_steps must be between 1 and 50")
    if guidance_scale <= 0.0 or guidance_scale > 20.0:
        raise HTTPException(status_code=400, detail="guidance_scale must be > 0 and <= 20")

    if not MODEL_DIR.exists():
        raise HTTPException(status_code=500, detail=f"Model dir not found: {MODEL_DIR}")
    # Allow first-run startup without local weights; some pipelines download
    # model artifacts lazily when checkpoint path is empty.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    descriptions = _load_descriptions()
    cloth_name = _safe_name(cloth_name)
    _sync_runtime_volume(commit=False)

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    person_filename, person_stem = _prepare_dataset_data(job_id, person_image, cloth_name, descriptions)

    async with inference_lock:
        try:
            output_dir, logs = await asyncio.to_thread(
                _run_inference,
                job_dir,
                person_filename,
                num_inference_steps,
                guidance_scale,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _build_tryon_response(job_id, output_dir, person_stem, cloth_name, logs)


@app.post("/api/tryon-json")
async def run_tryon_json(payload: TryOnJsonRequest):
    num_inference_steps = payload.num_inference_steps
    guidance_scale = payload.guidance_scale
    if num_inference_steps < 1 or num_inference_steps > 50:
        raise HTTPException(status_code=400, detail="num_inference_steps must be between 1 and 50")
    if guidance_scale <= 0.0 or guidance_scale > 20.0:
        raise HTTPException(status_code=400, detail="guidance_scale must be > 0 and <= 20")

    if not MODEL_DIR.exists():
        raise HTTPException(status_code=500, detail=f"Model dir not found: {MODEL_DIR}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    descriptions = _load_descriptions()
    cloth_name = _safe_name(payload.cloth_name)
    _sync_runtime_volume(commit=False)

    image_bytes = _decode_image_b64(payload.person_image_b64)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="person_image_b64 is empty")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    person_filename, person_stem = _prepare_dataset_data_from_bytes(
        job_id=job_id,
        person_image_bytes=image_bytes,
        person_image_filename=payload.person_image_filename,
        cloth_name=cloth_name,
        descriptions=descriptions,
    )

    async with inference_lock:
        try:
            output_dir, logs = await asyncio.to_thread(
                _run_inference,
                job_dir,
                person_filename,
                num_inference_steps,
                guidance_scale,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _build_tryon_response(job_id, output_dir, person_stem, cloth_name, logs)
