#!/usr/bin/env python3


# ─── CONFIG ──────────────────────────────────────────────────────────────────
SRC_ROOT       = "/scratch/ni00276/AML/Datasets/archive"          # contains G1020 ORIGA REFUGE
NNUNET_RAW     = "/scratch/ni00276/AML/U-Mamba/data/nnUNet_raw"   # nnUNet_raw root
DATASET_ID     = 202
DATASET_NAME   = "GlaucomaFundus_Processed"
TEST_SPLIT     = 0.10          # 0.10  →  10 % of cases go to imagesTs
RANDOM_SEED    = 42

APPLY_REMOVE_NERVES = True    # True/False
APPLY_CLAHE         = True    # True/False
# ─────────────────────────────────────────────────────────────────────────────

import json, random, shutil, sys
from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

random.seed(RANDOM_SEED)

# ─── helper: optional preprocessing ──────────────────────────────────────────
def remove_nerves(img_rgb: np.ndarray) -> np.ndarray:
    """Returns RGB float image in [0,1] with visible vessels in‑painted away."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, th = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpaint = cv2.inpaint(img_rgb, th, 1, cv2.INPAINT_TELEA)
    return inpaint

def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel in LAB space (keeps colour)."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

def preprocess(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    if APPLY_REMOVE_NERVES:
        arr = remove_nerves(arr)
    if APPLY_CLAHE:
        arr = apply_clahe(arr)
    return Image.fromarray(arr)

# ─── helper: masks ───────────────────────────────────────────────────────────
def discover_palette(mask_path: Path) -> Tuple[int, int | None]:
    vals = np.unique(np.array(Image.open(mask_path)))
    fg = sorted(int(v) for v in vals if v != 0)
    if not fg:
        raise RuntimeError(f"mask {mask_path} has no foreground")
    return (fg[0], fg[1]) if len(fg) > 1 else (fg[0], None)

def remap_mask(src: Path, dst: Path, disc: int, cup: int | None):
    arr = np.array(Image.open(src))
    out = np.zeros_like(arr, dtype=np.uint8)
    out[arr == disc] = 1
    if cup is not None:
        out[arr == cup] = 2
    Image.fromarray(out).save(dst)

# ─── gather files ────────────────────────────────────────────────────────────
DATASETS = ["G1020", "ORIGA", "REFUGE"]
IMAGE_DIRNAME = "Images_Square"
MASK_DIRNAME  = "Masks_Square"
IMAGE_EXTS    = (".png", ".jpg", ".jpeg")

def find_pairs() -> List[Tuple[Path, Path]]:
    pairs = []
    root = Path(SRC_ROOT)
    for ds in DATASETS:
        for img_dir in (root / ds).glob(f"**/{IMAGE_DIRNAME}"):
            mask_dir = img_dir.parent / MASK_DIRNAME
            if not mask_dir.is_dir():
                continue
            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                mask_path = mask_dir / f"{img_path.stem}.png"
                if mask_path.exists():
                    pairs.append((img_path, mask_path))
    return pairs

# ─── prepare dataset in required format ──────────────────────────────────────────────────────────────
def main():
    all_pairs = find_pairs()
    if not all_pairs:
        sys.exit("No image/mask pairs found ‑ check SRC_ROOT path.")
    random.shuffle(all_pairs)

    n_total = len(all_pairs)
    n_test  = int(round(TEST_SPLIT * n_total))
    test_pairs  = all_pairs[:n_test]
    train_pairs = all_pairs[n_test:]

    ds_dir   = Path(NNUNET_RAW) / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    imagesTr = ds_dir / "imagesTr"; imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr = ds_dir / "labelsTr"; labelsTr.mkdir(exist_ok=True)
    imagesTs = ds_dir / "imagesTs"; imagesTs.mkdir(exist_ok=True)

    case_idx = 0
    def save_case(img_path: Path, mask_path: Path, is_test: bool):
        nonlocal case_idx
        case = f"case{case_idx:04d}"
        dst_img = (imagesTs if is_test else imagesTr) / f"{case}_0000.png"
        img = preprocess(Image.open(img_path))
        img.save(dst_img, format="PNG")

        disc, cup = discover_palette(mask_path)
        dst_mask = (labelsTr if not is_test else None)
        if dst_mask:  # labelsTs may be empty
            remap_mask(mask_path, dst_mask / f"{case}.png", disc, cup)
        case_idx += 1

    for img_path, mask_path in tqdm(train_pairs, desc="train"):
        save_case(img_path, mask_path, is_test=False)
    for img_path, mask_path in tqdm(test_pairs, desc="test "):
        save_case(img_path, mask_path, is_test=True)

    # ─ dataset.json ─
    dataset_json = {
        "name":          DATASET_NAME,
        "tensorImageSize": "2D",
        "channel_names": { "0": "R", "1": "G", "2": "B" },
        "labels":        { "background": 0, "disc": 1, "cup": 2 },
        "numTraining":   len(train_pairs),
        "file_ending":   ".png"
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"\n✓ {len(train_pairs)} train  |  {len(test_pairs)} test  →  {ds_dir}")

if __name__ == "__main__":
    main()
