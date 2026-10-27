"""Image validation, preprocessing, and radiological enhancement for RadInsight AI.

Handles PNG and JPEG file formats, integrity verification, contrast enhancement (CLAHE),
and MobileNetV2 tensor preprocessing.
"""

import io
import logging
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config.settings import COLOR_CHANNELS, IMAGE_SIZE

logger = logging.getLogger(__name__)

                                   
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg"}

class ImageValidationError(Exception):
    """Raised when an uploaded image fails validation."""
    pass

def validate_image_file(file_input: Union[str, bytes, io.BytesIO, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate file format, integrity, dimensions, and readability.

    Args:
        file_input: File path, bytes, or file-like object (e.g. Streamlit UploadedFile).

    Returns:
        Tuple of (is_valid, message, metadata_dict)
    """
    metadata: Dict[str, Any] = {
        "filename": getattr(file_input, "name", "unknown_source"),
        "size_bytes": 0,
        "format": None,
        "mode": None,
        "width": 0,
        "height": 0,
    }

    try:
                                
        if isinstance(file_input, str):
            with open(file_input, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(file_input, bytes):
            raw_bytes = file_input
        elif hasattr(file_input, "read"):
                                 
            if hasattr(file_input, "seek"):
                file_input.seek(0)
            raw_bytes = file_input.read()
            if hasattr(file_input, "seek"):
                file_input.seek(0)
        else:
            return False, "Unsupported file input type provided.", metadata

        metadata["size_bytes"] = len(raw_bytes)

        if len(raw_bytes) == 0:
            return False, "Uploaded image file is empty (0 bytes).", metadata

                                    
        stream = io.BytesIO(raw_bytes)
        try:
            with Image.open(stream) as img:
                img.verify()                                  
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            return False, f"Corrupted or unrecognized image stream: {str(e)}", metadata

                                                                                
        stream.seek(0)
        with Image.open(stream) as img:
            format_name = (img.format or "").upper()
            metadata["format"] = format_name
            metadata["mode"] = img.mode
            metadata["width"] = img.width
            metadata["height"] = img.height

            if format_name not in ["PNG", "JPEG"]:
                return (
                    False,
                    f"Unsupported image format: {format_name}. Only standard PNG and JPEG chest X-rays are accepted.",
                    metadata,
                )

            if img.width < 64 or img.height < 64:
                return (
                    False,
                    f"Image resolution too low ({img.width}x{img.height}). Chest X-rays must be at least 64x64 pixels.",
                    metadata,
                )

        return True, "Image successfully validated.", metadata

    except Exception as e:
        logger.error(f"Image validation exception: {e}")
        return False, f"Unexpected validation error: {str(e)}", metadata

def load_image_to_pil(file_input: Union[str, bytes, io.BytesIO, Any]) -> Image.Image:
    """Load image from path or byte-stream into a standardized PIL RGB Image.

    Args:
        file_input: File path, bytes, or UploadedFile.

    Returns:
        PIL.Image.Image in RGB mode.
    """
    if isinstance(file_input, str):
        img = Image.open(file_input)
    elif isinstance(file_input, bytes):
        img = Image.open(io.BytesIO(file_input))
    elif hasattr(file_input, "read"):
        if hasattr(file_input, "seek"):
            file_input.seek(0)
        img = Image.open(file_input)
    else:
        raise ImageValidationError("Cannot convert input to PIL Image.")

                                    
    img = ImageOps.exif_transpose(img)

                                                                                
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img

def apply_clahe_enhancement(rgb_image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Standard clinical radiograph enhancement to accentuate fine parenchymal details,
    pleural margins, and retrocardiac lung fields.

    Args:
        rgb_image: uint8 numpy array with shape (H, W, 3).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        Enhanced RGB image as uint8 numpy array with shape (H, W, 3).
    """
                                                                                           
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)

    merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb

def preprocess_chest_xray(
    file_input: Union[str, bytes, io.BytesIO, Any],
    target_size: Tuple[int, int] = IMAGE_SIZE,
    enable_clahe: bool = True,
) -> Tuple[np.ndarray, Image.Image, Dict[str, Any]]:
    """Preprocess chest X-ray into MobileNetV2 compatible tensor format.

    Pipeline:
      1. Validation
      2. PIL load and RGB normalization
      3. High-quality bicubic resizing to target dimensions (224x224)
      4. Optional CLAHE radiological contrast enhancement
      5. Normalization to [-1.0, 1.0] matching MobileNetV2 standard input

    Args:
        file_input: Image path, bytes, or uploaded file.
        target_size: (height, width) tuple, defaults to (224, 224).
        enable_clahe: Whether to apply radiological CLAHE contrast enhancement.

    Returns:
        Tuple of (preprocessed_tensor_batch [1, 224, 224, 3], displayable_pil_image, metadata)
    """
    is_valid, msg, meta = validate_image_file(file_input)
    if not is_valid:
        raise ImageValidationError(msg)

                  
    pil_img = load_image_to_pil(file_input)

                                                 
    resized_pil = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(resized_pil, dtype=np.uint8)

                              
    if enable_clahe:
        enhanced_array = apply_clahe_enhancement(img_array)
        display_pil = Image.fromarray(enhanced_array)
    else:
        enhanced_array = img_array
        display_pil = resized_pil

                                                                         
                                
    normalized_tensor = (enhanced_array.astype(np.float32) / 127.5) - 1.0

                                                    
    tensor_batch = np.expand_dims(normalized_tensor, axis=0)

    meta["tensor_shape"] = list(tensor_batch.shape)
    meta["tensor_dtype"] = str(tensor_batch.dtype)
    meta["pixel_min"] = float(np.min(tensor_batch))
    meta["pixel_max"] = float(np.max(tensor_batch))
    meta["clahe_applied"] = enable_clahe

    return tensor_batch, display_pil, meta
