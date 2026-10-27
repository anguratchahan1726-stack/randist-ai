"""TensorFlow/Keras MobileNetV2 Deep Learning inference engine for RadInsight AI.

Constructs the multi-label chest radiography classifier, manages model checkpoints,
performs forward inference, and calculates raw scores with execution metadata.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config.settings import (
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_WEIGHTS_INFO_PATH,
    IMAGE_SIZE,
    SUPPORTED_FINDINGS,
)

logger = logging.getLogger(__name__)

                    
STATUS_FINE_TUNED = "FINE_TUNED"
STATUS_IMAGENET_BACKBONE = "IMAGENET_BACKBONE"
STATUS_UNTRAINED = "UNTRAINED"
STATUS_MISSING = "MISSING"

def build_mobilenetv2_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = len(SUPPORTED_FINDINGS),
    use_pretrained_backbone: bool = True,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """Construct a MobileNetV2-based multi-label chest radiograph classifier.

    Architecture:
      - Input Layer (224x224x3)
      - MobileNetV2 Feature Extractor (truncated before top classification head)
      - GlobalAveragePooling2D
      - BatchNormalization
      - Dropout (for regularizing transfer features)
      - Dense(128, activation='relu')
      - Dropout(0.2)
      - Dense(num_classes, activation='sigmoid')

    Multi-label classification uses Independent Sigmoid outputs because chest
    pathologies are non-mutually exclusive (e.g., Cardiomegaly + Pleural Effusion).

    Args:
        input_shape: Dimensions of input tensor.
        num_classes: Number of distinct findings.
        use_pretrained_backbone: Whether to load ImageNet weights for feature extraction.
        dropout_rate: Dropout probability for regularization.

    Returns:
        Compiled tf.keras.Model
    """
    weights = "imagenet" if use_pretrained_backbone else None

                                                        
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        alpha=1.0,                         
    )

                                                                 
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="chest_xray_input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="feature_dropout")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.2, name="classifier_dropout")(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="finding_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="RadInsight_MobileNetV2_Classifier")

                                                        
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    return model

class ChestXRayClassifier:
    """Wrapper managing MobileNetV2 model loading, metadata inspection, and inference."""

    _instance: Optional["ChestXRayClassifier"] = None

    def __init__(self, model_path: Optional[Path] = CLASSIFIER_MODEL_PATH):
        self.model_path = Path(model_path) if model_path else CLASSIFIER_MODEL_PATH
        self.findings_list: List[str] = list(SUPPORTED_FINDINGS)
        self.model: Optional[keras.Model] = None
        self.status: str = STATUS_MISSING
        self.metadata: Dict[str, Any] = {}
        self._load_or_initialize()

    @classmethod
    def get_instance(cls, model_path: Optional[Path] = None) -> "ChestXRayClassifier":
        """Singleton accessor for efficient memory utilization in Streamlit."""
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def _load_metadata(self) -> Dict[str, Any]:
        """Load external metadata JSON describing training history and checkpoint provenance."""
        if CLASSIFIER_WEIGHTS_INFO_PATH.exists():
            try:
                with open(CLASSIFIER_WEIGHTS_INFO_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not parse model metadata: {e}")
        return {
            "model_type": "MobileNetV2",
            "training_status": STATUS_IMAGENET_BACKBONE,
            "trained_epochs": 0,
            "dataset_source": "ImageNet initialization + Custom Chest Head",
            "notes": "Untrained / transfer initialization. Domain fine-tuning recommended on NIH ChestX-ray14.",
        }

    def _load_or_initialize(self) -> None:
        """Attempt loading trained checkpoint, or initialize architecture with ImageNet backbone."""
        meta = self._load_metadata()

        if self.model_path.exists():
            try:
                logger.info(f"Loading existing model checkpoint from {self.model_path}")
                self.model = keras.models.load_model(str(self.model_path), compile=False)
                self.metadata = meta
                self.status = meta.get("training_status", STATUS_FINE_TUNED)
                logger.info(f"Model loaded successfully with status: {self.status}")
                return
            except Exception as e:
                logger.error(f"Failed to load checkpoint at {self.model_path}: {e}")

                                                                                       
        logger.info("Initializing baseline MobileNetV2 architecture...")
        try:
            self.model = build_mobilenetv2_classifier(
                input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                num_classes=len(self.findings_list),
                use_pretrained_backbone=True,
            )
            self.status = STATUS_IMAGENET_BACKBONE
            self.metadata = meta
                                                                            
            self.save_checkpoint(self.model_path, status=STATUS_IMAGENET_BACKBONE)
        except Exception as e:
            logger.error(f"Failed to initialize MobileNetV2: {e}")
            self.model = None
            self.status = STATUS_MISSING

    def save_checkpoint(
        self,
        output_path: Path = CLASSIFIER_MODEL_PATH,
        status: str = STATUS_FINE_TUNED,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save Keras model checkpoint and accompanying metadata JSON."""
        if self.model is None:
            return False

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(output_path))

            meta = {
                "model_name": "RadInsight_MobileNetV2",
                "checkpoint_path": str(output_path),
                "num_classes": len(self.findings_list),
                "classes": self.findings_list,
                "training_status": status,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if extra_metadata:
                meta.update(extra_metadata)

            with open(CLASSIFIER_WEIGHTS_INFO_PATH, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            self.status = status
            self.metadata = meta
            logger.info(f"Model and metadata saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def predict(self, preprocessed_tensor: np.ndarray) -> Dict[str, Any]:
        """Perform forward inference on preprocessed radiograph tensor.

        Args:
            preprocessed_tensor: Tensor of shape (1, 224, 224, 3) normalized to [-1, 1].

        Returns:
            Dictionary with:
              - raw_scores: Dict[finding_name, score_float]
              - inference_time_ms: float
              - model_status: str (e.g. FINE_TUNED or IMAGENET_BACKBONE)
              - is_clinically_validated: bool (always False for academic research prototype)
        """
        if self.model is None:
            raise RuntimeError(
                "Model is not loaded. Ensure TensorFlow can initialize MobileNetV2."
            )

        start_time = time.perf_counter()
        raw_preds = self.model.predict(preprocessed_tensor, verbose=0)
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

                                                  
        scores_array = raw_preds[0]

        raw_scores: Dict[str, float] = {}
        for idx, finding_name in enumerate(self.findings_list):
            raw_scores[finding_name] = float(np.clip(scores_array[idx], 0.0, 1.0))

        return {
            "raw_scores": raw_scores,
            "inference_time_ms": round(inference_time_ms, 2),
            "model_status": self.status,
            "model_metadata": self.metadata,
            "is_clinically_validated": False,                                          
        }
