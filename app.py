"""
Flask Web Application for Medical Image Analysis
Provides user-friendly interface for analyzing chest X-rays and medical images.
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
import logging
import base64
import cv2
import numpy as np
from datetime import datetime
from medical_imaging_analyzer import MedicalImageAnalyzer, MedicalImagePreprocessor

# Configure Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'dcm'}

# Create upload folder
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize analyzer (use lighter model for web)
analyzer = MedicalImageAnalyzer(model_type="mobilenetv2")

# Disease risk levels
RISK_LEVELS = {
    "Normal": ("green", "Low Risk"),
    "Opacity": ("yellow", "Medium Risk"),
    "Pneumonia": ("orange", "High Risk"),
    "Tuberculosis": ("red", "Critical Risk"),
    "COVID-19": ("red", "Critical Risk")
}


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_path):
    """Convert image to base64 string for display."""
    try:
        with open(image_path, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        logger.error(f"Error converting image: {e}")
        return None


@app.route('/')
def index():
    """Render home page."""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """API endpoint for medical image analysis."""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, BMP'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        safe_filename = timestamp + filename
        filepath = Path(app.config['UPLOAD_FOLDER']) / safe_filename
        file.save(str(filepath))
        
        logger.info(f"Analyzing image: {safe_filename}")
        
        # Analyze image
        result = analyzer.analyze_image(str(filepath))
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        # Get disease and risk level
        disease = result.get('predicted_disease', 'Unknown')
        confidence = result.get('confidence', 0)
        risk_color, risk_level = RISK_LEVELS.get(disease, ("gray", "Unknown Risk"))
        
        # Convert image to base64
        img_b64 = image_to_base64(str(filepath))
        
        return jsonify({
            'success': True,
            'disease': disease,
            'confidence': confidence,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'all_predictions': result.get('all_predictions', {}),
            'image': img_b64,
            'filename': filename,
            'timestamp': result.get('timestamp', '')
        }), 200
    
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """API endpoint for batch analysis of multiple images."""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
                safe_filename = timestamp + filename
                filepath = Path(app.config['UPLOAD_FOLDER']) / safe_filename
                file.save(str(filepath))
                
                result = analyzer.analyze_image(str(filepath))
                
                if 'error' not in result:
                    disease = result.get('predicted_disease', 'Unknown')
                    confidence = result.get('confidence', 0)
                    risk_color, risk_level = RISK_LEVELS.get(disease, ("gray", "Unknown Risk"))
                    
                    results.append({
                        'filename': filename,
                        'disease': disease,
                        'confidence': confidence,
                        'risk_level': risk_level,
                        'risk_color': risk_color
                    })
        
        if not results:
            return jsonify({'error': 'No valid images processed'}), 400
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        }), 200
    
    except Exception as e:
        logger.error(f"Error during batch analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_type': 'mobilenetv2',
        'diseases': list(analyzer.DISEASE_CLASSES.values())
    }), 200


@app.route('/api/info', methods=['GET'])
def model_info():
    """Get model information."""
    return jsonify({
        'diseases': analyzer.DISEASE_CLASSES,
        'risk_levels': RISK_LEVELS,
        'max_file_size': '50MB',
        'supported_formats': list(ALLOWED_EXTENSIONS)
    }), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)
