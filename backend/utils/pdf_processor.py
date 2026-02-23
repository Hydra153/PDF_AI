"""
PDF & Image Processing Utilities
Handles PDF to image conversion using PyMuPDF (no OCR needed — Qwen2-VL reads pixels directly)
Also handles direct image uploads (PNG, JPG, JPEG, TIFF, BMP).
Applies image enhancement for scanned documents when enabled.
"""
import io
from typing import List, Dict, Tuple
from PIL import Image
import fitz  # PyMuPDF

from config import ENHANCE_SCANNED_IMAGES

# Render DPI for PDF → Image conversion
# 300 DPI is the OCR industry standard. Higher values get downscaled
# by the processor's smart_resize() anyway (max_pixels ≈ 1.3M).
DPI = 300

# Supported file extensions
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'}
SUPPORTED_PDF_EXTENSIONS = {'.pdf'}
SUPPORTED_EXTENSIONS = SUPPORTED_PDF_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS


def is_supported_file(filename: str) -> bool:
    """Check if filename has a supported extension (PDF or image)."""
    ext = '.' + filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in SUPPORTED_EXTENSIONS


def is_image_file(filename: str) -> bool:
    """Check if filename is an image (not PDF)."""
    ext = '.' + filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def pdf_to_images(pdf_file: bytes, dpi: int = DPI) -> List[Image.Image]:
    """
    Convert PDF bytes to list of PIL Images using PyMuPDF.
    
    Args:
        pdf_file: PDF file as bytes
        dpi: Resolution for conversion (default 300)
        
    Returns:
        List of PIL Image objects (one per page)
    """
    try:
        doc = fitz.open(stream=pdf_file, filetype="pdf")
        images = []
        
        for page in doc:
            # Set zoom factor based on DPI (72 dpi is default scale 1.0)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            
            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
            
        doc.close()
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []


def process_pdf(pdf_bytes: bytes) -> Tuple[List[Image.Image], List[Dict]]:
    """
    Process PDF: convert to images, optionally enhance for scanned documents.
    
    Returns:
        Tuple of (images, text_items)
        - images: List of PIL Images (enhanced if enabled)
        - text_items: Empty list (Qwen2-VL is OCR-free, no text extraction needed)
    """
    images = pdf_to_images(pdf_bytes)
    
    if not images:
        return [], []
    
    print(f"📄 Converted PDF to {len(images)} page image(s) at {DPI} DPI")
    
    # ── Image Enhancement for Scanned Documents ──
    if ENHANCE_SCANNED_IMAGES:
        from utils.image_enhancer import enhance_for_extraction
        enhanced_images = []
        for i, img in enumerate(images):
            print(f"   📄 Page {i+1}: analyzing scan quality...")
            enhanced_images.append(enhance_for_extraction(img))
        images = enhanced_images
    
    return images, []


def process_image(image_bytes: bytes) -> Tuple[List[Image.Image], List[Dict]]:
    """
    Process an uploaded image file: open with PIL, enhance.
    Same enhancement pipeline as PDFs but skips PDF→image conversion.
    
    Returns:
        Tuple of (images, text_items)
        - images: List containing the single enhanced PIL Image
        - text_items: Empty list
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if needed (handles RGBA, palette, grayscale)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        print(f"🖼️ Loaded image: {img.size[0]}x{img.size[1]} ({img.mode})")
        
        if ENHANCE_SCANNED_IMAGES:
            from utils.image_enhancer import enhance_for_extraction
            print(f"   🖼️ Analyzing image quality...")
            img = enhance_for_extraction(img)
        
        return [img], []
    except Exception as e:
        print(f"Error processing image: {e}")
        return [], []
