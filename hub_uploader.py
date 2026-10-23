import os
import logging
import threading
from pathlib import Path
from huggingface_hub import HfApi

logger = logging.getLogger("hub_uploader")

def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    tpath = Path.home() / ".cache" / "huggingface" / "token"
    if tpath.exists():
        try:
            return tpath.read_text().strip()
        except Exception:
            pass
    return ""

_upload_threads = []

def _worker(folder_path: str, repo_id: str, path_in_repo: str = None):
    try:
        token = get_hf_token()
        if not token:
            logger.error("No HF token found for upload")
            return
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=path_in_repo,
            token=token,
        )
        logger.info(f"Successfully uploaded {folder_path} to HF: {repo_id}/{path_in_repo or ''}")
    except Exception as e:
        logger.error(f"HF upload failed for {folder_path}: {e}")

def upload_folder_async(folder_path: str, repo_id: str, path_in_repo: str = None):
    p = Path(folder_path)
    if not p.exists():
        return
    t = threading.Thread(target=_worker, args=(str(folder_path), repo_id, path_in_repo), daemon=True)
    t.start()
    _upload_threads.append(t)
