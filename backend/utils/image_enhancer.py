"""
Image Enhancement for Document Extraction

Unified pipeline for ALL document types:
    1. Detect document type (clean digital vs scanned)
    2. Adaptive thresholding to create binary image
    3. Remove table borders (morphological line detection)
    4. Remove backgrounds and noise
    5. Produce clean black-text-on-white output

Inspired by Sarvam Vision: strip everything that isn't text
(backgrounds, borders, noise, colored cells) so the VLM sees
only the content that matters.
"""

import logging
import numpy as np
from PIL import Image, ImageStat

logger = logging.getLogger(__name__)

# Try OpenCV — fall back gracefully if unavailable
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    logger.warning("OpenCV not available — enhancement disabled")


def enhance_for_extraction(image: Image.Image) -> Image.Image:
    """
    Enhance a document image for VLM extraction.
    
    Applies to ALL documents:
        - Binarization (adaptive thresholding)
        - Table border removal (morphological opening)
        - Noise cleanup
    
    Args:
        image: PIL Image (any mode)
        
    Returns:
        Enhanced PIL Image in RGB mode — clean black text on white background
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    if not _HAS_CV2:
        print(f"   🖼️ OpenCV not available — returning original image")
        return image
    
    # ── Analyze image statistics ──
    grayscale = image.convert("L")
    stats = ImageStat.Stat(grayscale)
    mean_brightness = stats.mean[0]
    stddev = stats.stddev[0]
    
    is_dark = mean_brightness < 160
    
    print(f"   🖼️ Image stats: brightness={mean_brightness:.0f}, contrast={stddev:.0f}", end="")
    if is_dark:
        print(f" (dark scan)")
    elif mean_brightness > 230:
        print(f" (clean digital PDF)")
    else:
        print(f" (scanned document)")
    
    # ── Process with OpenCV ──
    enhanced = _clean_document(image, is_dark)
    
    # Final stats
    enhanced_gray = enhanced.convert("L")
    enhanced_stats = ImageStat.Stat(enhanced_gray)
    new_mean = enhanced_stats.mean[0]
    new_stddev = enhanced_stats.stddev[0]
    
    print(f"       ✅ Enhanced: brightness {mean_brightness:.0f}→{new_mean:.0f}, "
          f"contrast {stddev:.0f}→{new_stddev:.0f}")
    
    return enhanced


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Detect and correct document skew using Hough Line Transform.
    
    1. Detect edges with Canny
    2. Find lines with HoughLinesP
    3. Calculate median angle of near-horizontal lines
    4. Rotate image to correct skew (only if 0.3° < angle < 5°)
    
    Args:
        gray: Grayscale image (uint8)
        
    Returns:
        Deskewed grayscale image (or original if no skew detected)
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return gray  # No lines detected — skip
    
    # Collect angles of near-horizontal lines only
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 15:  # Within 15° of horizontal
            angles.append(angle)
    
    if not angles:
        return gray  # No near-horizontal lines found
    
    # Median is robust to outliers (stray diagonal lines)
    median_angle = float(np.median(angles))
    
    # Only correct meaningful skew (> 0.3°) but not extreme (< 5°)
    if abs(median_angle) < 0.3 or abs(median_angle) > 5.0:
        return gray
    
    print(f"       ✅ Deskew: correcting {median_angle:.1f}° rotation")
    
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=255)  # White fill
    return rotated


def _clean_document(image: Image.Image, is_dark: bool) -> Image.Image:
    """
    Universal document cleaning pipeline.
    
    Applied to ALL documents — clean digital PDFs and scans alike.
    
    Pipeline:
        1. Convert to grayscale
        2. CLAHE for dark scans (adaptive brightness normalization)
        3. Gaussian blur (noise reduction before thresholding)
        4. Adaptive Gaussian thresholding → binary black/white
        5. Detect and remove horizontal/vertical lines (table borders)
        6. Clean isolated noise specks
        7. Convert back to RGB
    """
    # PIL → NumPy
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # ── Step 0: Deskew (correct scan rotation) ──
    gray = _deskew(gray)
    
    # ── Step 1: CLAHE for dark scans ──
    if is_dark:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        print(f"       ✅ CLAHE applied (dark scan brightened)")
    
    # ── Step 2: Gaussian blur ──
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # ── Step 3: Adaptive Gaussian thresholding ──
    # Converts ALL backgrounds (gray, colored, white) to pure white
    # Text stays black regardless of original background color
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=51,       # Large neighborhood for smooth thresholding
        C=10,               # Offset — tuned to preserve text in headers
    )
    print(f"       ✅ Adaptive thresholding (backgrounds → white)")
    
    # ── Step 4: Remove table borders ──
    binary = _remove_lines(binary)
    print(f"       ✅ Table borders removed")
    
    # ── Step 5: Clean isolated noise specks ──
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_kernel)
    
    # ── Step 6: Smart crop (trim white margins) ──
    binary = _smart_crop(binary)
    
    # Convert back to RGB
    result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(result)


def _remove_lines(binary: np.ndarray) -> np.ndarray:
    """
    Detect and remove horizontal + vertical lines (table borders).
    
    Uses morphological opening with rectangular kernels to find lines
    longer than a minimum length, then whites them out.
    
    Args:
        binary: Binary (thresholded) grayscale image
        
    Returns:
        Binary image with lines removed
    """
    # Invert: lines become white (foreground) for morphological detection
    inverted = cv2.bitwise_not(binary)
    
    # Detect horizontal lines (min 60px long, 1px thick)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    h_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, h_kernel)
    
    # Detect vertical lines (min 60px tall, 1px thick)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
    v_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, v_kernel)
    
    # Dilate detected lines slightly to cover any anti-aliasing
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    h_lines = cv2.dilate(h_lines, dilate_kernel, iterations=1)
    v_lines = cv2.dilate(v_lines, dilate_kernel, iterations=1)
    
    # Combine all detected lines
    all_lines = cv2.add(h_lines, v_lines)
    
    # White out the lines in the original binary image
    result = cv2.add(binary, all_lines)
    
    return result


def _smart_crop(binary: np.ndarray, padding: int = 20) -> np.ndarray:
    """
    Trim white margins from binarized document.
    
    Finds the bounding box of all non-white (text) content,
    then crops with padding to preserve document edges.
    Only crops if margins exceed 5% of the image area.
    
    Args:
        binary: Binary image (white bg, black text)
        padding: Pixels of white space to preserve around content
        
    Returns:
        Cropped binary image (or original if margins are negligible)
    """
    # Invert so text becomes white (foreground)
    inverted = cv2.bitwise_not(binary)
    
    # Find all non-zero (text) pixel coordinates
    coords = cv2.findNonZero(inverted)
    
    if coords is None:
        return binary  # Entirely blank page
    
    # Bounding rectangle of all text content
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add padding (clamped to image boundaries)
    img_h, img_w = binary.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    
    # Only crop if we'd remove meaningful margin (> 5% of area)
    cropped_area = (x2 - x1) * (y2 - y1)
    original_area = img_w * img_h
    margin_ratio = 1.0 - (cropped_area / original_area)
    
    if margin_ratio < 0.05:
        return binary  # Margins too small to bother
    
    print(f"       ✅ Smart crop: removed {margin_ratio:.0%} empty margins "
          f"({img_w}x{img_h} → {x2-x1}x{y2-y1})")
    return binary[y1:y2, x1:x2]
