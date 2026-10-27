"""Centralized configuration for RadInsight AI.

Defines global parameters, finding definitions, thresholds, image preprocessing
dimensions, and path specifications.
"""

from pathlib import Path
from typing import Dict, List

                  
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TESTS_DIR = BASE_DIR / "tests"

                                                                                                 
SUPPORTED_FINDINGS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Infiltration",
    "Pleural Effusion",
    "Pneumothorax",
    "No Finding",
]

                                                                                       
FINDING_DESCRIPTIONS: Dict[str, str] = {
    "Atelectasis": "Partial collapse or volume loss of lung tissue with opacification.",
    "Cardiomegaly": "Enlargement of cardiac silhouette (cardiothoracic ratio > 0.50).",
    "Consolidation": "Alveolar airspace filling with fluid, pus, or cellular debris.",
    "Edema": "Diffuse haziness or perihilar cuffing indicative of fluid overload.",
    "Infiltration": "Ill-defined parenchymal density without frank volume loss.",
    "Pleural Effusion": "Blunting of the costophrenic sulcus secondary to pleural fluid accumulation.",
    "Pneumothorax": "Presence of pleural line with absent peripheral lung markings.",
    "No Finding": "Clear lung fields, sharp costophrenic angles, and normal cardiothoracic anatomy.",
}

                                       
ANATOMICAL_LOCATIONS: Dict[str, str] = {
    "Atelectasis": "lung bases / retrocardiac regions",
    "Cardiomegaly": "cardiac silhouette",
    "Consolidation": "pulmonary airspaces",
    "Edema": "perihilar and interstitial zones",
    "Infiltration": "pulmonary parenchyma",
    "Pleural Effusion": "costophrenic angles / pleural spaces",
    "Pneumothorax": "pleural cavity / apical zones",
    "No Finding": "bilateral hemithoraces",
}

                             
IMAGE_SIZE = (224, 224)
COLOR_CHANNELS = 3
NORMALIZATION_RANGE = (-1.0, 1.0)                                            

                                     
                                                                                             
                                                                       
DEFAULT_DECISION_THRESHOLDS: Dict[str, float] = {
    "Atelectasis": 0.45,
    "Cardiomegaly": 0.50,
    "Consolidation": 0.45,
    "Edema": 0.45,
    "Infiltration": 0.45,
    "Pleural Effusion": 0.45,
    "Pneumothorax": 0.35,                                                   
    "No Finding": 0.55,
}

                                                                                    
UNCERTAINTY_MARGIN = 0.12

                      
CLASSIFIER_MODEL_PATH = MODELS_DIR / "mobilenetv2_chest_xray.keras"
CLASSIFIER_WEIGHTS_INFO_PATH = MODELS_DIR / "model_metadata.json"

                                    
REPORT_MODEL_NAME = "google/flan-t5-small"
REPORT_MAX_LENGTH = 256

                                                           
THEME_COLORS = {
    "primary": "#1e3a8a",                  
    "secondary": "#0d9488",                 
    "accent": "#0284c7",                  
    "background": "#f8fafc",                 
    "card_bg": "#ffffff",                   
    "text_main": "#0f172a",                           
    "text_muted": "#475569",               
    "border": "#e2e8f0",                           
    "status_positive": "#15803d",
    "status_warning": "#b45309",
    "status_negative": "#b91c1c",
}
