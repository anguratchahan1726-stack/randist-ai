"""Generates demonstration radiological chest images and ground-truth evaluation manifest.

Simulates authentic chest radiograph silhouettes (rib cage, bilateral lung fields,
mediastinum, diaphragmatic domes, costophrenic sulci) and pathology patterns.
"""

import json
import sys
from pathlib import Path

                          
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image

from config.settings import DATA_DIR

SAMPLE_DIR = DATA_DIR / "sample_test"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

def generate_base_radiograph(width: int = 512, height: int = 512) -> np.ndarray:
    """Generate anatomically structured radiograph background."""
    img = np.zeros((height, width), dtype=np.uint8)

                                                   
    cv2.ellipse(img, (width // 2, height // 2), (width // 2 - 20, height // 2 - 30), 0, 0, 360, 40, -1)

                                                                   
                                               
    cv2.ellipse(img, (width // 2 - 110, height // 2 - 10), (80, 160), -5, 0, 360, 180, -1)
                                              
    cv2.ellipse(img, (width // 2 + 110, height // 2 - 10), (75, 155), 5, 0, 360, 180, -1)

                                                              
    cv2.rectangle(img, (width // 2 - 35, 40), (width // 2 + 35, height - 80), 90, -1)

               
    cv2.line(img, (60, 110), (width // 2 - 20, 140), 160, 10)
    cv2.line(img, (width - 60, 110), (width // 2 + 20, 140), 160, 10)

                           
    for y_offset in range(160, height - 120, 38):
        cv2.ellipse(img, (width // 2 - 110, y_offset), (78, 20), -8, 0, 180, 120, 4)
        cv2.ellipse(img, (width // 2 + 110, y_offset), (75, 20), 8, 0, 180, 120, 4)

                         
    cv2.ellipse(img, (width // 2 - 110, height - 110), (85, 50), 0, 180, 360, 80, -1)
    cv2.ellipse(img, (width // 2 + 110, height - 100), (80, 45), 0, 180, 360, 80, -1)

                                                                                
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    noise = np.random.normal(0, 8, (height, width)).astype(np.int16)
    noisy_img = np.clip(blurred.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                                                                        
    radiograph = 255 - noisy_img
    return radiograph

def create_all_samples():
    manifest_records = []

                              
    img1 = generate_base_radiograph()
                                           
    cv2.ellipse(img1, (256 + 25, 330), (55, 65), 30, 0, 360, 210, -1)
    img1_smooth = cv2.GaussianBlur(img1, (7, 7), 0)
    path1 = SAMPLE_DIR / "normal_cxr_01.png"
    Image.fromarray(img1_smooth).save(path1)
    manifest_records.append({
        "image_file": "sample_test/normal_cxr_01.png",
        "case_id": "CXR-1001",
        "patient_view": "PA",
        "labels": ["No Finding"]
    })

                                                                 
    img2 = generate_base_radiograph()
    cv2.ellipse(img2, (256 + 55, 340), (105, 95), 30, 0, 360, 225, -1)
    img2_smooth = cv2.GaussianBlur(img2, (7, 7), 0)
    path2 = SAMPLE_DIR / "cardiomegaly_cxr_02.png"
    Image.fromarray(img2_smooth).save(path2)
    manifest_records.append({
        "image_file": "sample_test/cardiomegaly_cxr_02.png",
        "case_id": "CXR-1002",
        "patient_view": "AP",
        "labels": ["Cardiomegaly"]
    })

                                                                 
    img3 = generate_base_radiograph()
                                                                        
    cv2.polygon = np.array([[60, 440], [170, 440], [180, 480], [60, 480]], dtype=np.int32)
    cv2.fillPoly(img3, [cv2.polygon], 220)
    cv2.ellipse(img3, (256 + 25, 330), (55, 65), 30, 0, 360, 210, -1)
    img3_smooth = cv2.GaussianBlur(img3, (9, 9), 0)
    path3 = SAMPLE_DIR / "pleural_effusion_cxr_03.png"
    Image.fromarray(img3_smooth).save(path3)
    manifest_records.append({
        "image_file": "sample_test/pleural_effusion_cxr_03.png",
        "case_id": "CXR-1003",
        "patient_view": "PA",
        "labels": ["Pleural Effusion"]
    })

                                                                           
    img4 = generate_base_radiograph()
                            
    cv2.ellipse(img4, (160, 260), (45, 55), 15, 0, 360, 220, -1)
    cv2.ellipse(img4, (256 + 25, 330), (55, 65), 30, 0, 360, 210, -1)
    img4_smooth = cv2.GaussianBlur(img4, (11, 11), 0)
    path4 = SAMPLE_DIR / "consolidation_cxr_04.png"
    Image.fromarray(img4_smooth).save(path4)
    manifest_records.append({
        "image_file": "sample_test/consolidation_cxr_04.png",
        "case_id": "CXR-1004",
        "patient_view": "PA",
        "labels": ["Consolidation", "Infiltration"]
    })

                                                                     
    img5 = generate_base_radiograph()
                                                             
    cv2.ellipse(img5, (380, 140), (25, 65), 10, 200, 340, 240, 2)
    cv2.ellipse(img5, (256 + 25, 330), (55, 65), 30, 0, 360, 210, -1)
    img5_smooth = cv2.GaussianBlur(img5, (5, 5), 0)
    path5 = SAMPLE_DIR / "pneumothorax_cxr_05.png"
    Image.fromarray(img5_smooth).save(path5)
    manifest_records.append({
        "image_file": "sample_test/pneumothorax_cxr_05.png",
        "case_id": "CXR-1005",
        "patient_view": "PA",
        "labels": ["Pneumothorax"]
    })

                                
    img6 = generate_base_radiograph()
    cv2.ellipse(img6, (256 + 25, 330), (55, 65), 30, 0, 360, 210, -1)
    img6_smooth = cv2.GaussianBlur(img6, (7, 7), 0)
    path6 = SAMPLE_DIR / "normal_cxr_06.jpg"
    Image.fromarray(img6_smooth).save(path6, "JPEG", quality=95)
    manifest_records.append({
        "image_file": "sample_test/normal_cxr_06.jpg",
        "case_id": "CXR-1006",
        "patient_view": "PA",
        "labels": ["No Finding"]
    })

                   
    manifest_file = DATA_DIR / "ground_truth.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)

    print(f"Generated 6 sample chest radiographs in: {SAMPLE_DIR}")
    print(f"Saved evaluation ground truth manifest to: {manifest_file}")

if __name__ == "__main__":
    create_all_samples()
