import argparse
import logging
import os
import sys
import time
from pathlib import Path


os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")  
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")       
os.environ.setdefault("FASTEMBED_CACHE_PATH", os.path.join(os.environ["HF_HOME"], "fastembed"))
os.environ.setdefault("TQDM_MININTERVAL", "5")           

logging.basicConfig(level=logging.INFO, format="%(asctime)s [start] %(message)s")
log = logging.getLogger("kognit.start")

DENSE_MODEL = os.getenv("DENSE_MODEL") or "BAAI/bge-large-en-v1.5"
SPARSE_MODEL = os.getenv("SPARSE_MODEL") or "Qdrant/bm25"
RERANKER_MODEL = os.getenv("RERANKER_MODEL") or "BAAI/bge-reranker-large"

MAX_ATTEMPTS = int(os.getenv("MODEL_PREFETCH_ATTEMPTS", "20"))
FORCE_VERIFY = os.getenv("FORCE_MODEL_VERIFY") == "1"
MARKER_DIR = Path(os.environ["HF_HOME"]) / ".kognit"


ALWAYS_IGNORE = [
    "onnx/*", "openvino/*", "coreml/*", "*.onnx", "*.onnx_data", "*.ot",
    "*.msgpack", "*.h5", "*.tflite", "*.gguf", "*.mlmodel", "*.mlpackage/*",
]


def with_retries(label, fn):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as exc: 
            if attempt == MAX_ATTEMPTS:
                log.error("%s: giving up after %d attempts: %r", label, attempt, exc)
                raise
            wait = min(5 * 2 ** (attempt - 1), 60)
            log.warning("%s: attempt %d/%d failed (%r). Resuming in %ds...",
                        label, attempt, MAX_ATTEMPTS, exc, wait)
            time.sleep(wait)


def _marker(name: str) -> Path:
    return MARKER_DIR / (name.replace("/", "--") + ".ok")


def _has_weights(snapshot: Path) -> bool:
    files = list(snapshot.glob("*.safetensors")) + list(snapshot.glob("pytorch_model*.bin"))
    return any(p.exists() for p in files)


def _cached_snapshot(repo_id: str):
    repo_dir = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + repo_id.replace("/", "--"))
    snaps = repo_dir / "snapshots"
    ref = repo_dir / "refs" / "main"
    if ref.exists():
        cand = snaps / ref.read_text().strip()
        if cand.is_dir():
            return cand
    if snaps.is_dir():
        dirs = sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if dirs:
            return dirs[0]
    return None


def prefetch_hf_repo(repo_id: str) -> None:
    from huggingface_hub import HfApi, snapshot_download

    snap = _cached_snapshot(repo_id)
    complete = snap is not None and _has_weights(snap)

    if complete and not FORCE_VERIFY and _marker(repo_id).exists():
        log.info("[ok] %s already complete in cache - skipping download", repo_id)
        return
    if _marker(repo_id).exists() and not complete:
        log.warning("%s: marker present but cache incomplete (snapshot=%s); re-downloading",
                    repo_id, snap)

    def _download():
        files = HfApi().list_repo_files(repo_id)
        ignore = list(ALWAYS_IGNORE)
        if any("/" not in f and f.endswith(".safetensors") for f in files):
            ignore += ["pytorch_model*.bin"]
        log.info("Downloading %s (resumable)...", repo_id)
        return snapshot_download(repo_id, ignore_patterns=ignore, max_workers=2)

    try:
        path = Path(with_retries(repo_id, _download))
    except Exception:
        if complete:
            log.warning("%s: Hub unreachable, but a cached copy with weights exists - "
                        "continuing from cache", repo_id)
            return
        raise

    if not _has_weights(path):
        raise RuntimeError(f"{repo_id}: download finished but no weight file found in {path}")

    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    _marker(repo_id).write_text(str(path))
    log.info("[ok] %s ready", repo_id)


def prefetch_fastembed(model_name: str) -> None:
    """fastembed keeps its own cache layout (FASTEMBED_CACHE_PATH)."""
    if not FORCE_VERIFY and _marker("fastembed-" + model_name).exists():
        log.info("[ok] %s (fastembed) already in cache - skipping download", model_name)
        return

    def _download():
        from fastembed import SparseTextEmbedding
        log.info("Downloading %s via fastembed into %s ...",
                 model_name, os.environ["FASTEMBED_CACHE_PATH"])
        SparseTextEmbedding(model_name=model_name)

    with_retries(model_name, _download)
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    _marker("fastembed-" + model_name).write_text("ok")
    log.info("[ok] %s ready", model_name)


def prefetch() -> None:
    log.info("HF_HOME=%s  FASTEMBED_CACHE_PATH=%s  xet_disabled=%s",
             os.environ["HF_HOME"], os.environ["FASTEMBED_CACHE_PATH"],
             os.environ["HF_HUB_DISABLE_XET"])
    prefetch_fastembed(SPARSE_MODEL)      
    prefetch_hf_repo(DENSE_MODEL)         
    prefetch_hf_repo(RERANKER_MODEL)      
    log.info("All models present in cache.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefetch-only", action="store_true")
    args = parser.parse_args()

    if os.getenv("SKIP_MODEL_PREFETCH") != "1":
        try:
            prefetch()
        except Exception as exc: 
            log.error("Model prefetch failed: %r", exc)
            log.error("Exiting non-zero; Docker will restart and the download resumes "
                      "from the partial files already in the cache volume.")
            return 1

    if args.prefetch_only:
        return 0

    
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    log.info("Starting API (models loaded from local cache, HF offline mode ON)")
    os.execv(
        sys.executable,
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "0.0.0.0", "--port", "8000"],
    )


if __name__ == "__main__":
    sys.exit(main())