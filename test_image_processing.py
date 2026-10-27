"""Unit tests for image validation and preprocessing module."""

import io
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from PIL import Image

from image_processing import (
    ImageValidationError,
    apply_clahe_enhancement,
    preprocess_chest_xray,
    validate_image_file,
)


def create_dummy_png(width: int = 128, height: int = 128, color: int = 120) -> bytes:
    """Helper creating memory-buffered PNG image."""
    img = Image.new("L", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_dummy_jpeg(width: int = 128, height: int = 128) -> bytes:
    """Helper creating memory-buffered JPEG image."""
    img = Image.new("RGB", (width, height), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_validate_valid_png():
    raw_bytes = create_dummy_png(200, 200)
    is_valid, msg, meta = validate_image_file(raw_bytes)
    assert is_valid is True
    assert meta["format"] == "PNG"
    assert meta["width"] == 200
    assert meta["height"] == 200


def test_validate_valid_jpeg():
    raw_bytes = create_dummy_jpeg(256, 256)
    is_valid, msg, meta = validate_image_file(raw_bytes)
    assert is_valid is True
    assert meta["format"] == "JPEG"


def test_validate_empty_bytes():
    is_valid, msg, meta = validate_image_file(b"")
    assert is_valid is False
    assert "empty" in msg.lower()


def test_validate_corrupted_stream():
    corrupted = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRbad_stream"
    is_valid, msg, meta = validate_image_file(corrupted)
    assert is_valid is False
    assert "corrupted" in msg.lower() or "unrecognized" in msg.lower()


def test_validate_low_resolution():
    tiny = create_dummy_png(32, 32)
    is_valid, msg, meta = validate_image_file(tiny)
    assert is_valid is False
    assert "too low" in msg.lower()


def test_preprocess_chest_xray_output():
    raw_bytes = create_dummy_png(256, 256)
    tensor_batch, display_pil, meta = preprocess_chest_xray(raw_bytes, enable_clahe=True)

    # Check tensor shape and properties
    assert tensor_batch.shape == (1, 224, 224, 3)
    assert tensor_batch.dtype == np.float32

    # Check normalization range [-1.0, 1.0]
    assert np.min(tensor_batch) >= -1.0
    assert np.max(tensor_batch) <= 1.0

    # Check display image properties
    assert isinstance(display_pil, Image.Image)
    assert display_pil.size == (224, 224)
    assert meta["clahe_applied"] is True


def test_clahe_enhancement_shape():
    sample_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    enhanced = apply_clahe_enhancement(sample_array)
    assert enhanced.shape == sample_array.shape
    assert enhanced.dtype == np.uint8
