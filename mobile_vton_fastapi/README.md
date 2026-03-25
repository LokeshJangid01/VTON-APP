# Mobile-VTON FastAPI Wrapper

This service runs `2026_CVPR_Mobile-VTON/inference.py` from a web API and serves a simple HTML UI.

## 1) Activate your conda env

```bash
cd /d D:\Mobile-VTON\2026_CVPR_Mobile-VTON
conda activate D:\Mobile-VTON\2026_CVPR_Mobile-VTON\.conda\envs\mobile_cpu
```

## 2) Install API dependencies

```bash
cd /d D:\Mobile-VTON\mobile_vton_fastapi
python -m pip install -r requirements.txt
```

## 3) Start server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open: `http://127.0.0.1:8000`

## Notes

- Cloth list comes from `D:\Mobile-VTON\2026_CVPR_Mobile-VTON\Mobile_VTON\VITON-HD\test\cloth`.
- Each request creates a per-job runtime folder under `mobile_vton_fastapi/runtime/jobs`.
- API runs one inference at a time to avoid overlapping model runs.

