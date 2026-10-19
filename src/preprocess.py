#!/usr/bin/env python3
"""
OpenCV Switchable Pre-processing Module
Provides modular, switchable image preprocessing stages for Bengali OCR:
  - Greyscale conversion
  - Median blur denoising
  - CLAHE contrast enhancement
  - Otsu binarisation
  - Robust document deskewing
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import yaml


def to_greyscale(img: np.ndarray) -> np.ndarray:
    """Convert image to single-channel greyscale if not already greyscale."""
    if img is None:
        raise ValueError("Input image is None")
    if len(img.shape) == 2:
        return img.copy()
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def apply_median_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply median blur denoising. ksize must be an odd integer >= 1."""
    if img is None:
        raise ValueError("Input image is None")
    if ksize <= 1:
        return img.copy()
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img, ksize)


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Works on greyscale directly, or on the L channel of LAB for color images.
    """
    if img is None:
        raise ValueError("Input image is None")

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))

    if len(img.shape) == 2:
        return clahe.apply(img)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        merged = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    else:
        # Fallback to greyscale CLAHE
        gray = to_greyscale(img)
        return clahe.apply(gray)


def apply_otsu_binarisation(img: np.ndarray) -> np.ndarray:
    """Convert image to pure binary (0 and 255) using Otsu's thresholding."""
    if img is None:
        raise ValueError("Input image is None")
    gray = to_greyscale(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def estimate_skew_angle(
    img: np.ndarray,
    max_angle_deg: float = 10.0,
    min_angle_deg: float = 0.2
) -> float:
    """
    Estimate page skew angle using Hough line transform on text edges.
    Returns angle in degrees (positive = clockwise, negative = counter-clockwise).
    Returns 0.0 if skew is negligible or outside reasonable bounds.
    """
    gray = to_greyscale(img)

    # Downscale for fast and robust skew estimation
    h, w = gray.shape[:2]
    scale = 1000.0 / max(h, w) if max(h, w) > 1000 else 1.0
    if scale < 1.0:
        resized = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = gray

    # Edge detection
    edges = cv2.Canny(resized, 50, 150, apertureSize=3)

    # Detect line segments using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=int(resized.shape[1] * 0.2),
        maxLineGap=20
    )

    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # Keep angles near horizontal (-45 to 45 degrees)
        if abs(angle) <= max_angle_deg:
            angles.append(angle)

    if not angles:
        return 0.0

    # Use median angle to reject outliers
    median_angle = float(np.median(angles))

    if abs(median_angle) < min_angle_deg:
        return 0.0

    return median_angle


def apply_deskew(
    img: np.ndarray,
    max_angle_deg: float = 10.0,
    min_angle_deg: float = 0.2,
    background_color: Tuple[int, ...] = (255, 255, 255)
) -> np.ndarray:
    """
    Deskew document image based on estimated skew angle.
    If angle is below min_angle_deg or exceeds max_angle_deg, returns original image unchanged.
    """
    if img is None:
        raise ValueError("Input image is None")

    angle = estimate_skew_angle(img, max_angle_deg=max_angle_deg, min_angle_deg=min_angle_deg)
    if abs(angle) < min_angle_deg:
        return img.copy()

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Determine border fill based on channels
    border_val = 255 if len(img.shape) == 2 else (255, 255, 255)

    deskewed = cv2.warpAffine(
        img,
        rot_mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val
    )
    return deskewed


def load_preprocessing_config(config_path: Union[str, Path] = "configs/config.yaml") -> dict:
    """Load preprocessing settings from YAML config."""
    p = Path(config_path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("preprocessing", {})


def preprocess_image(
    img_or_path: Union[str, Path, np.ndarray],
    config: Optional[dict] = None
) -> np.ndarray:
    """
    Master pipeline function that sequentially executes enabled preprocessing steps.

    If mode is 'raw' or config is empty/all disabled, returns the loaded image without change.
    """
    # 1. Load image
    if isinstance(img_or_path, (str, Path)):
        img_path = str(img_or_path)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image from: {img_path}")
    elif isinstance(img_or_path, np.ndarray):
        img = img_or_path.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(img_or_path)}")

    # 2. Resolve configuration
    if config is None:
        config = load_preprocessing_config()

    mode = config.get("mode", "custom")
    if mode == "raw":
        return img

    # 3. Sequential pipeline
    out = img

    # Deskew (best done on original image before binarisation)
    deskew_cfg = config.get("deskew", {})
    if isinstance(deskew_cfg, dict) and deskew_cfg.get("enabled", False):
        max_a = deskew_cfg.get("max_angle_deg", 10.0)
        out = apply_deskew(out, max_angle_deg=max_a)

    # Greyscale
    if config.get("to_greyscale", False):
        out = to_greyscale(out)

    # Median blur
    blur_cfg = config.get("median_blur", {})
    if isinstance(blur_cfg, dict) and blur_cfg.get("enabled", False):
        ksize = blur_cfg.get("ksize", 3)
        out = apply_median_blur(out, ksize=ksize)

    # CLAHE
    clahe_cfg = config.get("clahe", {})
    if isinstance(clahe_cfg, dict) and clahe_cfg.get("enabled", False):
        clip = clahe_cfg.get("clip_limit", 2.0)
        grid = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
        out = apply_clahe(out, clip_limit=clip, tile_grid_size=grid)

    # Otsu binarisation
    otsu_cfg = config.get("otsu_binarisation", {})
    is_otsu = (otsu_cfg is True) or (isinstance(otsu_cfg, dict) and otsu_cfg.get("enabled", False))
    if is_otsu:
        out = apply_otsu_binarisation(out)

    return out


def preprocess_split(
    pages_csv_path: Union[str, Path] = "data/pages.csv",
    split: str = "dev",
    output_dir: Union[str, Path] = "data/preprocessed_scans/dev",
    page_ids: Optional[List[str]] = None,
    allow_test: bool = False,
    config_path: Union[str, Path] = "configs/config.yaml"
) -> List[Dict[str, Any]]:
    """
    Process images for a dataset split under the full OCR preprocessing pipeline:
    greyscale -> median_blur -> clahe -> otsu -> deskew.
    Saves processed images to output_dir and returns structured execution logs.
    Strictly guards the held-out test set if allow_test=False.
    """
    # Guard held-out test split
    if split == "test" and not allow_test:
        cfg = {}
        if Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        if cfg.get("test_set_read_only", True):
            raise PermissionError(
                "Held-out test set protection is ACTIVE! "
                "Running preprocessing on 'test' split is prohibited during development. "
                "Pass allow_test=True explicitly for final test execution."
            )

    csv_p = Path(pages_csv_path)
    if not csv_p.exists():
        raise FileNotFoundError(f"Pages manifest not found at: {csv_p}")

    import pandas as pd
    df = pd.read_csv(csv_p)
    target_df = df[(df["split"] == split) & (df["page_type"] == "core_prose")]

    if page_ids:
        target_df = target_df[target_df["page_id"].isin(page_ids)]

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    results = []
    for _, row in target_df.iterrows():
        p_id = str(row["page_id"])
        img_p = Path(str(row["image_path"]))
        if not img_p.exists():
            continue

        raw_img = cv2.imread(str(img_p))
        if raw_img is None:
            continue

        h, w = raw_img.shape[:2]
        skew_angle = estimate_skew_angle(raw_img)

        # Full preprocessing pipeline:
        # 1. Greyscale
        gray = to_greyscale(raw_img)
        # 2. Median blur
        blurred = apply_median_blur(gray, ksize=3)
        # 3. CLAHE
        clahe_img = apply_clahe(blurred, clip_limit=2.0, tile_grid_size=(8, 8))
        # 4. Otsu binarisation
        binary = apply_otsu_binarisation(clahe_img)
        # 5. Deskew
        deskewed = apply_deskew(binary, max_angle_deg=10.0)

        dest_file = out_p / f"{p_id}.png"
        cv2.imwrite(str(dest_file), deskewed)

        rec = {
            "page_id": p_id,
            "book_page": int(row["book_page"]),
            "width": w,
            "height": h,
            "detected_skew_deg": round(skew_angle, 4),
            "output_path": str(dest_file),
            "file_size_bytes": dest_file.stat().st_size
        }
        results.append(rec)

    return results

