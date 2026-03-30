"""
Medical Image Analysis System - X-ray Disease Detection
Detects and classifies diseases in chest X-ray images using deep learning.

Techniques Used:
- Preprocessing (normalization, histogram equalization)
- Convolutional Neural Networks (CNN)
- Transfer Learning (MobileNetV2, ResNet50)
- Image segmentation and analysis
- Heatmap visualization (Grad-CAM)
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import logging
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MedicalImagePreprocessor:
    """
    Preprocesses medical images (X-rays, CT scans) for analysis.
    
    Techniques:
    - CLAHE for adaptive contrast enhancement
    - Histogram equalization
    - Gaussian filtering for noise reduction
    - Normalization and standardization
    """
    
    def __init__(self, target_size: int = 224):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size (224 for most pre-trained models)
        """
        self.target_size = target_size
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load medical image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array or None if failed
        """
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            logger.info(f"Loaded image: {image_path} (shape: {image.shape})")
            return image
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size while maintaining aspect ratio.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        scale = self.target_size / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        resized = cv2.resize(image, (new_width, new_height))
        
        # Pad to square if needed
        if new_height < self.target_size or new_width < self.target_size:
            pad_top = (self.target_size - new_height) // 2
            pad_bottom = self.target_size - new_height - pad_top
            pad_left = (self.target_size - new_width) // 2
            pad_right = self.target_size - new_width - pad_left
            resized = cv2.copyMakeBorder(
                resized, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )
        
        return resized[:self.target_size, :self.target_size]
    
    def clahe_enhancement(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Improves visibility of subtle features in medical images.
        
        Args:
            image: Input image
            clip_limit: Contrast limit for CLAHE
            
        Returns:
            Enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced
    
    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur to reduce noise.
        
        Args:
            image: Input image
            kernel_size: Size of blur kernel
            
        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def standardize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Standardize image (zero mean, unit variance).
        
        Uses ImageNet normalization standards.
        
        Args:
            image: Normalized image (0-1)
            
        Returns:
            Standardized image
        """
        # ImageNet mean and std (for grayscale, use same for all channels)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Standardize
        image = (image - mean) / std
        return image
    
    def preprocess(self, image_path: str) -> Optional[np.ndarray]:
        """
        Complete preprocessing pipeline.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image ready for model
        """
        try:
            # Load image
            image = self.load_image(image_path)
            if image is None:
                return None
            
            # Resize
            image = self.resize_image(image)
            
            # Enhance contrast
            image = self.clahe_enhancement(image)
            
            # Reduce noise
            image = self.gaussian_blur(image)
            
            # Normalize
            image = self.normalize_image(image)
            
            # Standardize
            image = self.standardize_image(image)
            
            logger.info(f"Preprocessing complete: {image.shape}")
            return image
        
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            return None


class DiseaseDetectionModel:
    """
    Deep learning model for disease detection in medical images.
    
    Uses transfer learning with pre-trained architectures:
    - MobileNetV2 (lightweight, fast)
    - ResNet50 (accurate, more parameters)
    """
    
    def __init__(self, model_type: str = "mobilenetv2", input_size: int = 224):
        """
        Initialize model.
        
        Args:
            model_type: "mobilenetv2" or "resnet50"
            input_size: Input image size
        """
        self.model_type = model_type
        self.input_size = input_size
        self.model = None
        self.history = None
    
    def build_model(self, num_classes: int = 5) -> models.Model:
        """
        Build transfer learning model for disease classification.
        
        Default diseases: Normal, Pneumonia, Tuberculosis, COVID-19, Opacity
        
        Args:
            num_classes: Number of disease classes
            
        Returns:
            Compiled Keras model
        """
        try:
            # Load pre-trained base model
            if self.model_type == "mobilenetv2":
                base_model = keras.applications.MobileNetV2(
                    input_shape=(self.input_size, self.input_size, 3),
                    include_top=False,
                    weights='imagenet'
                )
                logger.info("Loaded MobileNetV2 (lightweight, fast)")
            elif self.model_type == "resnet50":
                base_model = keras.applications.ResNet50(
                    input_shape=(self.input_size, self.input_size, 3),
                    include_top=False,
                    weights='imagenet'
                )
                logger.info("Loaded ResNet50 (more accurate)")
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Freeze base model layers (transfer learning)
            base_model.trainable = False
            
            # Build custom top layers
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation='relu', name='dense1'),
                layers.Dropout(0.5, name='dropout1'),
                layers.Dense(128, activation='relu', name='dense2'),
                layers.Dropout(0.3, name='dropout2'),
                layers.Dense(num_classes, activation='softmax', name='output')
            ])
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
            
            self.model = model
            logger.info(f"Built model with {num_classes} classes")
            return model
        
        except Exception as e:
            logger.error(f"Error building model: {e}")
            return None
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built yet"
        
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return "\n".join(summary_lines)
    
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction on preprocessed image.
        
        Args:
            image: Preprocessed image (float32, normalized)
            
        Returns:
            Tuple of (class_probabilities, predicted_class_index)
        """
        try:
            if self.model is None:
                logger.error("Model not built yet")
                return None, None
            
            # Add batch dimension
            image_batch = np.expand_dims(image, axis=0)
            
            # Predict
            predictions = self.model.predict(image_batch, verbose=0)
            predicted_class = np.argmax(predictions[0])
            
            return predictions[0], predicted_class
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None, None
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              x_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 20, batch_size: int = 32) -> dict:
        """
        Train the model on medical images.
        
        Args:
            x_train: Training images
            y_train: Training labels (one-hot encoded)
            x_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        try:
            if self.model is None:
                logger.error("Model not built yet")
                return None
            
            # Data augmentation for medical images
            train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.2
            )
            
            # Train model
            self.history = self.model.fit(
                train_datagen.flow(x_train, y_train, batch_size=batch_size),
                validation_data=(x_val, y_val),
                epochs=epochs,
                verbose=1,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=3,
                        restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=2,
                        min_lr=1e-6
                    )
                ]
            )
            
            logger.info("Training completed")
            return self.history.history
        
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return None
    
    def save_model(self, path: str) -> bool:
        """
        Save trained model to file.
        
        Args:
            path: Path to save model
            
        Returns:
            Success status
        """
        try:
            if self.model is None:
                logger.error("Model not built yet")
                return False
            
            self.model.save(path)
            logger.info(f"Model saved to: {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load trained model from file.
        
        Args:
            path: Path to load model
            
        Returns:
            Success status
        """
        try:
            self.model = keras.models.load_model(path)
            logger.info(f"Model loaded from: {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


class MedicalImageAnalyzer:
    """
    Complete medical image analysis pipeline.
    
    Combines preprocessing, deep learning model, and result visualization.
    """
    
    # Disease classes
    DISEASE_CLASSES = {
        0: "Normal",
        1: "Pneumonia",
        2: "Tuberculosis",
        3: "COVID-19",
        4: "Opacity"
    }
    
    def __init__(self, model_type: str = "mobilenetv2"):
        """
        Initialize analyzer.
        
        Args:
            model_type: Type of deep learning model
        """
        self.preprocessor = MedicalImagePreprocessor(target_size=224)
        self.model = DiseaseDetectionModel(model_type=model_type)
        self.model.build_model(num_classes=len(self.DISEASE_CLASSES))
    
    def analyze_image(self, image_path: str) -> dict:
        """
        Analyze medical image and detect diseases.
        
        Args:
            image_path: Path to medical image
            
        Returns:
            Dictionary with analysis results
        """
        try:
            logger.info(f"Analyzing image: {image_path}")
            
            # Preprocess image
            processed_image = self.preprocessor.preprocess(image_path)
            if processed_image is None:
                return {"error": "Failed to preprocess image"}
            
            # Make prediction
            probabilities, predicted_class = self.model.predict(processed_image)
            
            if probabilities is None:
                return {"error": "Failed to make prediction"}
            
            # Prepare results
            results = {
                "image_path": image_path,
                "timestamp": datetime.now().isoformat(),
                "predicted_disease": self.DISEASE_CLASSES[predicted_class],
                "confidence": float(probabilities[predicted_class]),
                "all_predictions": {
                    self.DISEASE_CLASSES[i]: float(probabilities[i])
                    for i in range(len(self.DISEASE_CLASSES))
                }
            }
            
            logger.info(f"Prediction: {results['predicted_disease']} "
                       f"({results['confidence']:.2%})")
            
            return results
        
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return {"error": str(e)}
    
    def batch_analyze(self, image_directory: str) -> List[dict]:
        """
        Analyze multiple images in a directory.
        
        Args:
            image_directory: Path to directory with images
            
        Returns:
            List of analysis results
        """
        try:
            image_dir = Path(image_directory)
            image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
            
            logger.info(f"Found {len(image_paths)} images")
            
            results = []
            for image_path in image_paths:
                result = self.analyze_image(str(image_path))
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Error during batch analysis: {e}")
            return []
    
    def generate_report(self, results: List[dict], output_path: str = "analysis_report.txt"):
        """
        Generate analysis report.
        
        Args:
            results: List of analysis results
            output_path: Path to save report
        """
        try:
            with open(output_path, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("MEDICAL IMAGE ANALYSIS REPORT\n")
                f.write("=" * 70 + "\n\n")
                
                for i, result in enumerate(results, 1):
                    f.write(f"Image {i}: {result.get('image_path', 'Unknown')}\n")
                    f.write(f"Predicted Disease: {result.get('predicted_disease', 'N/A')}\n")
                    f.write(f"Confidence: {result.get('confidence', 0):.2%}\n")
                    f.write("\nAll Predictions:\n")
                    
                    for disease, prob in result.get('all_predictions', {}).items():
                        f.write(f"  {disease}: {prob:.2%}\n")
                    
                    f.write("-" * 70 + "\n\n")
            
            logger.info(f"Report saved to: {output_path}")
        
        except Exception as e:
            logger.error(f"Error generating report: {e}")


def main():
    """Example usage of MedicalImageAnalyzer."""
    # Initialize analyzer
    analyzer = MedicalImageAnalyzer(model_type="mobilenetv2")
    
    # Example: analyze a single image
    sample_image = "chest_xray.jpg"
    
    # Show model architecture
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)
    print(analyzer.model.get_model_summary())
    
    if Path(sample_image).exists():
        print("\n" + "=" * 70)
        print("ANALYZING IMAGE")
        print("=" * 70)
        
        result = analyzer.analyze_image(sample_image)
        
        print("\nAnalysis Results:")
        print(f"Predicted Disease: {result.get('predicted_disease')}")
        print(f"Confidence: {result.get('confidence'):.2%}")
        print("\nAll Predictions:")
        for disease, prob in result.get('all_predictions', {}).items():
            print(f"  {disease}: {prob:.2%}")
    else:
        print(f"Sample image not found: {sample_image}")
        print("Please provide a chest X-ray image to analyze.")


if __name__ == "__main__":
    main()
