import base64
import os
import subprocess
import tempfile
import time

import requests
import runpod
import torch
from transformers import pipeline

MODEL_ID = os.environ["MODEL_ID"]
LANGUAGE = os.environ.get("LANGUAGE", "yoruba")
HF_TOKEN = os.environ.get("HF_TOKEN") or None            # only needed for private models
FORCE_LANGUAGE = os.environ.get("FORCE_LANGUAGE", "1") == "1"
MAX_DOWNLOAD_MB = int(os.environ.get("MAX_DOWNLOAD_MB", "500"))

print(f"Loading {MODEL_ID} ({LANGUAGE}) ...")
_t0 = time.time()
asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    torch_dtype=torch.float16,
    device="cuda:0",
    chunk_length_s=30,          # handles long audio
    token=HF_TOKEN,
)
print(f"Model loaded in {time.time() - _t0:.1f}s")


def _download(url: str, dest: str) -> None:
    limit = MAX_DOWNLOAD_MB * 1024 * 1024
    size = 0
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise ValueError(f"Audio exceeds {MAX_DOWNLOAD_MB} MB limit")
                f.write(chunk)


def _duration(path: str):
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            stderr=subprocess.DEVNULL, timeout=30,
        )
        return round(float(out.strip()), 2)
    except Exception:
        return None


def handler(job):
    """
    Input:
      audio_url     : URL to an audio file  (or)
      audio_base64  : base64-encoded audio bytes
      return_segments : default true
    The language is fixed per endpoint via the LANGUAGE env var.
    """
    inp = job.get("input", {})
    url = inp.get("audio_url")
    b64 = inp.get("audio_base64")
    if not url and not b64:
        return {"error": "Provide 'audio_url' or 'audio_base64'."}

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
            tmp_path = tmp.name

        if url:
            _download(url, tmp_path)
        else:
            with open(tmp_path, "wb") as f:
                f.write(base64.b64decode(b64))

        t0 = time.time()
        gen_kwargs = {"task": "transcribe"}
        if FORCE_LANGUAGE:
            gen_kwargs["language"] = LANGUAGE

        result = asr(tmp_path, return_timestamps=True, generate_kwargs=gen_kwargs)

        segments = []
        for c in result.get("chunks", []):
            start, end = c["timestamp"]
            segments.append({
                "start": round(start, 2) if start is not None else None,
                "end": round(end, 2) if end is not None else None,
                "text": c["text"].strip(),
            })

        out = {
            "text": result["text"].strip(),
            "language": LANGUAGE,
            "audio_duration": _duration(tmp_path),
            "processing_time": round(time.time() - t0, 2),
            "model": MODEL_ID,
        }
        if inp.get("return_segments", True):
            out["segments"] = segments
        return out

    except Exception as e:
        return {"error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


runpod.serverless.start({"handler": handler})
