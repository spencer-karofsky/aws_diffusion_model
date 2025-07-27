# scripts/prepare_midjourney_dataset_parallel.py
import os, math, time, requests, pandas as pd
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple

def _safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def _split_2x2(img: Image.Image, trim_px: int = 0) -> List[Image.Image]:
    w, h = img.size
    w2, h2 = w // 2, h // 2
    boxes = [
        (0,    0,   w2,  h2),  # TL
        (w2,   0,   w,   h2),  # TR
        (0,    h2,  w2,  h),   # BL
        (w2,   h2,  w,   h),   # BR
    ]
    if trim_px > 0:
        boxes = [
            (max(0, x0 + (trim_px if i in (1,3) else 0)),
             max(0, y0 + (trim_px if i in (2,3) else 0)),
             min(w, x1 - (trim_px if i in (0,2) else 0)),
             min(h, y1 - (trim_px if i in (0,1) else 0)))
            for i, (x0, y0, x1, y1) in enumerate(boxes)
        ]
    return [img.crop(b) for b in boxes]

def _download_bytes(url: str, timeout: int, max_retries: int = 4, backoff: float = 0.75) -> bytes:
    last_err = None
    for attempt in range(max_retries):
        try:
            with requests.get(url, timeout=timeout) as r:
                r.raise_for_status()
                return r.content
        except Exception as e:
            last_err = e
            time.sleep(backoff * (2 ** attempt))
    raise last_err

def _process_row(
    idx: int,
    prompt: str,
    url: str,
    images_dir: str,
    mode: str,
    resize_save: Optional[int],
    trim_px: int,
    timeout: int
) -> List[Dict[str, str]]:
    """
    Returns list of rows [{'caption':..., 'image_path':...}, ...] or [] on failure.
    """
    try:
        content = _download_bytes(url, timeout=timeout)
        grid = Image.open(BytesIO(content)).convert("RGB")
    except Exception as e:
        # swallow error; caller will skip
        return []

    crops: List[Image.Image]
    if mode == "split":
        crops = _split_2x2(grid, trim_px=trim_px)
    else:
        w2, h2 = grid.width // 2, grid.height // 2
        crops = [grid.crop((0, 0, w2, h2))]

    out_rows: List[Dict[str, str]] = []
    for j, crop in enumerate(crops):
        if resize_save is not None:
            crop = crop.resize((resize_save, resize_save), Image.LANCZOS)

        fname = f"mj_{idx:07d}_{j}.jpg" if mode == "split" else f"mj_{idx:07d}_0.jpg"
        path = os.path.join(images_dir, fname)
        if not os.path.exists(path):
            crop.save(path, format="JPEG", quality=95)
        out_rows.append({"caption": prompt, "image_path": path})

    return out_rows

def prepare_midjourney_local_dataset_parallel(
    parquet_path: str = "hf://datasets/CortexLM/midjourney-v6/data/train-00000-of-00001.parquet",
    out_dir: str = "dalle2/data/local_datasets/midjourney_v6",
    max_grids: int = 10_000,
    mode: str = "split",            # "split" -> 4 crops, "single" -> keep top-left only
    resize_save: int = 256,         # save 256x256 for storage efficiency
    trim_px: int = 2,
    timeout: int = 15,
    max_workers: int = 24           # tune: 16–32 is usually good
) -> str:
    assert mode in ("split", "single")
    images_dir = os.path.join(out_dir, "images")
    _safe_mkdir(images_dir)
    _safe_mkdir(out_dir)

    df = pd.read_parquet(parquet_path).head(max_grids)

    tasks = []
    results: List[Dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, row in df.iterrows():
            prompt = row["prompt"]
            url = row["image_url"]
            fut = ex.submit(
                _process_row,
                i, prompt, url,
                images_dir, mode, resize_save, trim_px, timeout
            )
            tasks.append(fut)

        with tqdm(total=len(tasks), desc="Downloading & splitting", unit="grid") as pbar:
            for fut in as_completed(tasks):
                rows = fut.result()
                if rows:
                    results.extend(rows)
                pbar.update(1)

    meta = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "metadata.csv")
    meta.to_csv(csv_path, index=False)
    print(f"[ok] wrote {csv_path} with {len(meta)} rows")
    return csv_path

if __name__ == "__main__":
    prepare_midjourney_local_dataset_parallel()
