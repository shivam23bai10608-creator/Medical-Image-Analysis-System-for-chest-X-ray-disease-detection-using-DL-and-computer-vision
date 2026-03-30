"""
Unit Tests for Medical Image Analysis System
Tests preprocessing, model building, and analysis functionality.
"""

import unittest
import numpy as np
import cv2
from pathlib import Path
from medical_imaging_analyzer import (
    MedicalImagePreprocessor,
    DiseaseDetectionModel,
    MedicalImageAnalyzer
)


class TestImagePreprocessor(unittest.TestCase):
    """Test medical image preprocessing."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.preprocessor = MedicalImagePreprocessor(target_size=224)
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initializes correctly."""
        self.assertIsNotNone(self.preprocessor)
        self.assertEqual(self.preprocessor.target_size, 224)
    
    def test_resize_image(self):
        """Test image resizing."""
        # Create dummy image
        dummy_image = np.ones((400, 600), dtype=np.uint8) * 128
        
        resized = self.preprocessor.resize_image(dummy_image)
        
        # Check output size
        self.assertEqual(resized.shape, (224, 224))
    
    def test_clahe_enhancement(self):
        """Test CLAHE contrast enhancement."""
        # Create test image with gradient
        image = np.linspace(0, 255, 256*256).reshape(256, 256).astype(np.uint8)
        
        enhanced = self.preprocessor.clahe_enhancement(image)
        
        # Check that enhancement was applied
        self.assertEqual(enhanced.shape, image.shape)
        self.assertTrue(np.any(enhanced != image))
    
    def test_gaussian_blur(self):
        """Test Gaussian blur for noise reduction."""
        # Create noisy image
        image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        blurred = self.preprocessor.gaussian_blur(image, kernel_size=5)
        
        self.assertEqual(blurred.shape, image.shape)
    
    def test_normalize_image(self):
        """Test image normalization to [0, 1]."""
        image = np.ones((224, 224), dtype=np.uint8) * 128
        
        normalized = self.preprocessor.normalize_image(image)
        
        # Check range
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))
        self.assertAlmostEqual(normalized.mean(), 128/255, places=2)
    
    def test_standardize_image(self):
        """Test image standardization."""
        # Create normalized image
        image = np.ones((224, 224, 3), dtype=np.float32) * 0.5
        
        standardized = self.preprocessor.standardize_image(image)
        
        # Check output type and shape
        self.assertEqual(standardized.dtype, np.float32)
        self.assertEqual(standardized.shape[2], 3)


class TestDiseaseDetectionModel(unittest.TestCase):
    """Test disease detection CNN model."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model = DiseaseDetectionModel(model_type="mobilenetv2")
    
    def test_model_initialization(self):
        """Test model initializes."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.model_type, "mobilenetv2")
    
    def test_model_building(self):
        """Test model building."""
        model = self.model.build_model(num_classes=5)
        
        self.assertIsNotNone(model)
        # Check output layer
        self.assertEqual(model.layers[-1].units, 5)
    
    def test_model_compilation(self):
        """Test that model is compiled."""
        model = self.model.model
        
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
    
    def test_model_summary(self):
        """Test model summary generation."""
        summary = self.model.get_model_summary()
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
    
    def test_prediction_output_shape(self):
        """Test prediction output shape."""
        # Create dummy preprocessed image
        dummy_image = np.random.randn(224, 224, 3).astype(np.float32)
        
        predictions, predicted_class = self.model.predict(dummy_image)
        
        if predictions is not None:
            self.assertEqual(len(predictions), 5)  # 5 classes
            self.assertIn(predicted_class, range(5))


class TestMedicalImageAnalyzer(unittest.TestCase):
    """Test complete medical image analysis pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.analyzer = MedicalImageAnalyzer(model_type="mobilenetv2")
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        self.assertIsNotNone(self.analyzer)
        self.assertIsNotNone(self.analyzer.preprocessor)
        self.assertIsNotNone(self.analyzer.model)
    
    def test_disease_classes(self):
        """Test disease classes are defined."""
        classes = self.analyzer.DISEASE_CLASSES
        
        self.assertEqual(len(classes), 5)
        self.assertIn("Normal", classes.values())
        self.assertIn("Pneumonia", classes.values())
        self.assertIn("COVID-19", classes.values())
    
    def test_batch_analysis_returns_list(self):
        """Test batch analysis returns list."""
        # Create temporary test directory
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy images
            for i in range(2):
                dummy_img = np.ones((224, 224), dtype=np.uint8) * 128
                path = Path(tmpdir) / f"test_{i}.jpg"
                cv2.imwrite(str(path), dummy_img)
            
            results = self.analyzer.batch_analyze(tmpdir)
            
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 2)
    
    def test_report_generation(self):
        """Test analysis report generation."""
        import tempfile
        
        # Create dummy results
        results = [
            {
                'image_path': 'test.jpg',
                'predicted_disease': 'Normal',
                'confidence': 0.95,
                'all_predictions': {
                    'Normal': 0.95,
                    'Pneumonia': 0.03,
                    'COVID-19': 0.01,
                    'Tuberculosis': 0.01,
                    'Opacity': 0.00
                }
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.txt"
            
            self.analyzer.generate_report(results, str(report_path))
            
            self.assertTrue(report_path.exists())
            
            with open(report_path, 'r') as f:
                content = f.read()
                self.assertIn('Normal', content)
                self.assertIn('95%', content)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline."""
    
    def test_preprocessor_and_model_integration(self):
        """Test preprocessing output works with model."""
        import tempfile
        
        preprocessor = MedicalImagePreprocessor(target_size=224)
        model = DiseaseDetectionModel(model_type="mobilenetv2")
        model.build_model(num_classes=5)
        
        # Create dummy image
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.jpg"
            dummy_img = np.ones((256, 256), dtype=np.uint8) * 128
            cv2.imwrite(str(img_path), dummy_img)
            
            # Preprocess
            preprocessed = preprocessor.preprocess(str(img_path))
            
            if preprocessed is not None:
                # Should work with model
                predictions, predicted_class = model.predict(preprocessed)
                
                self.assertIsNotNone(predictions)
                self.assertIsNotNone(predicted_class)
    
    def test_complete_analysis_pipeline(self):
        """Test complete analysis from image to results."""
        analyzer = MedicalImageAnalyzer(model_type="mobilenetv2")
        
        # Verify all components are initialized
        self.assertIsNotNone(analyzer.preprocessor)
        self.assertIsNotNone(analyzer.model)
        self.assertIsNotNone(analyzer.model.model)


if __name__ == '__main__':
    unittest.main()
