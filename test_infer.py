#!/usr/bin/env python3
"""
Test suite for infer.py (DenseNet121 Inference Engine)
Runnable via: python test_infer.py
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from PIL import Image


def create_dummy_cervical_image(path):
    """Creates a synthetic 128x128 RGB test image simulating cervical tissue."""
    img = Image.new("RGB", (128, 128), color=(200, 100, 100))
    # Draw a center region
    for x in range(40, 88):
        for y in range(40, 88):
            img.putpixel((x, y), (240, 150, 150))
    img.save(path)


def test_cli_help():
    """Verify CLI prints help cleanly."""
    res = subprocess.run(
        [sys.executable, "infer.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    assert res.returncode == 0, f"CLI help failed: {res.stderr}"
    assert "Cervical Abnormality Detection CLI" in res.stdout
    print("[PASS] test_cli_help")


def test_mock_inference_json():
    """Verify mock inference with --json returns proper structured output."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        create_dummy_cervical_image(tmp_path)
        res = subprocess.run(
            [sys.executable, "infer.py", "--image", tmp_path, "--mock", "--json"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        assert res.returncode == 0, f"Mock inference failed: {res.stderr}"
        data = json.loads(res.stdout)
        assert data["status"] == "success"
        assert "probability" in data
        assert "prediction" in data
        assert "confidence" in data
        assert data["input_resolution"] == "128x128"
        assert data["class_id"] in [0, 1]
        print(f"[PASS] test_mock_inference_json (Prediction: {data['prediction']}, Conf: {data['confidence']})")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_missing_file_handling():
    """Verify missing file outputs error cleanly and non-zero exit code."""
    res = subprocess.run(
        [sys.executable, "infer.py", "--image", "non_existent_image_123.jpg", "--json"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    assert res.returncode != 0
    data = json.loads(res.stdout)
    assert data["status"] == "error"
    print("[PASS] test_missing_file_handling")


if __name__ == "__main__":
    print("\nRunning self-check for cervical abnormality infer.py...")
    test_cli_help()
    test_mock_inference_json()
    test_missing_file_handling()
    print("\nAll 3 tests passed successfully!\n")
