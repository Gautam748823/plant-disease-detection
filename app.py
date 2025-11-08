"""
Plant Disease Detection Web Application
Main Flask application for detecting plant diseases from leaf images
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load model and class labels
# Use absolute path to avoid file not found errors
MODEL_PATH = 'C:/Users/gauta/plant-disease-detection/models/plant_disease_model_compatible.h5'
LABELS_PATH = 'C:/Users/gauta/plant-disease-detection/models/class_labels.json'

print("Loading model...")
try:
    # Try loading with compile=False to avoid compatibility issues
    model = load_model(MODEL_PATH, compile=False)
    
    # Recompile with current TensorFlow version
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nTrying alternative loading method...")
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Model loaded with alternative method!")
    except Exception as e2:
        print(f"❌ Alternative method failed: {e2}")
        print("\n⚠️ IMPORTANT: Your model may have been trained with a different TensorFlow version.")
        print("Please retrain the model with TensorFlow 2.20 or downgrade to TensorFlow 2.15")
        exit(1)

print("Loading class labels...")
with open(LABELS_PATH, 'r') as f:
    class_labels = json.load(f)
print(f"✅ {len(class_labels)} classes loaded!")

# Disease information and treatment recommendations
# This will match ANY disease name format
def get_disease_info_smart(disease_name):
    """Smart disease info lookup that handles different naming formats"""
    
    # Normalize the disease name - remove all special characters
    normalized = disease_name.lower()
    normalized = normalized.replace('___', ' ').replace('_', ' ').replace('-', ' ').replace(',', '')
    normalized = '_'.join(normalized.split())  # Convert spaces to underscores
    
    print(f"🔍 Looking up disease: '{disease_name}' -> normalized: '{normalized}'")  # Debug
    
    # Comprehensive disease database
    disease_db = {
        # Tomato diseases
        "tomato_bacterial_spot": {
            "description": "Bacterial spot is caused by Xanthomonas bacteria and appears as dark spots on leaves and fruits. It thrives in warm, humid conditions.",
            "treatment": "Remove infected leaves immediately, apply copper-based fungicides, avoid overhead watering, ensure good air circulation between plants.",
            "prevention": "Use disease-free seeds, practice crop rotation (3-4 years), maintain proper spacing, avoid working with wet plants."
        },
        "tomato_early_blight": {
            "description": "Early blight is a fungal disease causing brown spots with concentric rings (target-like) on older leaves. Can reduce yield significantly.",
            "treatment": "Remove infected leaves, apply fungicides containing chlorothalonil or copper, improve air circulation, mulch to prevent soil splash.",
            "prevention": "Use resistant varieties, mulch around plants, water at soil level, practice crop rotation, remove plant debris."
        },
        "tomato_late_blight": {
            "description": "Late blight is a devastating disease that can destroy entire crops in days. Caused by the same pathogen that caused Irish potato famine.",
            "treatment": "Remove and destroy infected plants IMMEDIATELY, apply protective fungicides, harvest unaffected fruits early, improve drainage.",
            "prevention": "Plant resistant varieties, ensure good drainage, monitor weather conditions, use certified disease-free transplants."
        },
        "tomato_leaf_mold": {
            "description": "Leaf mold creates yellow spots on upper leaf surfaces with olive-green to grayish-purple fuzzy growth underneath. Common in greenhouses.",
            "treatment": "Improve ventilation significantly, reduce humidity below 85%, apply fungicides if severe, remove infected leaves.",
            "prevention": "Use resistant varieties, maintain proper spacing (2-3 feet), avoid overhead irrigation, ensure greenhouse ventilation."
        },
        "tomato_septoria_leaf_spot": {
            "description": "Septoria leaf spot causes small circular spots with dark borders and gray centers on leaves. Spreads rapidly in wet conditions.",
            "treatment": "Remove infected leaves, apply copper-based fungicides weekly, mulch to prevent splash, avoid overhead watering.",
            "prevention": "Practice crop rotation, use disease-free transplants, maintain proper spacing, remove plant debris after harvest."
        },
        "tomato_spider_mites": {
            "description": "Spider mites are tiny pests (barely visible) that cause yellowing, stippling, and webbing on leaves. Thrive in hot, dry conditions.",
            "treatment": "Spray with strong water stream daily, apply insecticidal soap or neem oil, introduce predatory mites, increase humidity.",
            "prevention": "Maintain adequate moisture, introduce predatory mites early, avoid dusty conditions, monitor regularly."
        },
        "tomato_target_spot": {
            "description": "Target spot creates brown lesions with concentric rings on leaves and fruits. Can cause significant yield loss if untreated.",
            "treatment": "Remove infected leaves immediately, apply fungicides (azoxystrobin or chlorothalonil), improve air circulation.",
            "prevention": "Use resistant varieties, practice crop rotation, avoid overhead watering, maintain plant spacing."
        },
        "tomato_yellow_leaf_curl_virus": {
            "description": "Viral disease transmitted by whiteflies causing severe leaf curling, yellowing, and stunted growth. No cure available.",
            "treatment": "Remove infected plants immediately, control whitefly population aggressively, use reflective mulches, plant virus-free transplants.",
            "prevention": "Use virus-free transplants, control whiteflies with yellow sticky traps, use resistant varieties, use insect-proof screens."
        },
        "tomato_mosaic_virus": {
            "description": "Mosaic virus causes mottled yellow-green patterns on leaves, leaf distortion, and reduced fruit quality. Spreads through handling.",
            "treatment": "Remove and destroy infected plants, disinfect all tools with bleach, control aphids, wash hands thoroughly before handling plants.",
            "prevention": "Use virus-free seeds, practice good sanitation, control insect vectors, avoid tobacco use near plants."
        },
        "tomato_healthy": {
            "description": "Your tomato plant appears healthy with vibrant green leaves and no visible signs of disease! Continue your excellent care.",
            "treatment": "Continue regular care: consistent watering, balanced fertilization, and monitoring for pests.",
            "prevention": "Maintain current practices: proper watering, fertilization, pruning, and regular inspection for early problem detection."
        },
        
        # Potato diseases
        "potato_early_blight": {
            "description": "Early blight causes dark brown spots with concentric rings on potato leaves. Can reduce tuber size and quality significantly.",
            "treatment": "Apply fungicides (chlorothalonil, mancozeb), remove infected foliage, ensure proper spacing for air flow, hill soil around plants.",
            "prevention": "Use certified seed potatoes, practice 3-4 year crop rotation, maintain soil health, avoid overhead irrigation."
        },
        "potato_late_blight": {
            "description": "Late blight is a serious disease that destroyed Irish potato crops in 1840s. Can destroy entire fields rapidly in cool, wet weather.",
            "treatment": "Apply protective fungicides immediately, destroy infected plants, harvest early if weather favors disease, improve drainage.",
            "prevention": "Plant resistant varieties, monitor weather closely, ensure good drainage, use certified disease-free seed potatoes."
        },
        "potato_healthy": {
            "description": "Your potato plant is healthy and thriving! Leaves are green and vigorous with no disease symptoms visible.",
            "treatment": "Continue regular maintenance: consistent watering, hilling, and pest monitoring.",
            "prevention": "Maintain good agricultural practices: proper spacing, regular inspection, and timely harvesting."
        },
        
        # Pepper diseases
        "pepper_bell_bacterial_spot": {
            "description": "Bacterial spot causes dark, greasy-looking lesions on pepper leaves, stems, and fruits. Spreads rapidly in warm, wet conditions.",
            "treatment": "Apply copper-based bactericides, remove infected leaves, improve air circulation, avoid working with wet plants.",
            "prevention": "Use disease-free seeds and transplants, practice crop rotation, avoid overhead watering, maintain plant spacing."
        },
        "pepper_bell_healthy": {
            "description": "Your bell pepper plant is healthy and thriving! Continue your excellent care routine for best fruit production.",
            "treatment": "Continue regular maintenance: consistent watering, balanced fertilization, and support for branches with fruits.",
            "prevention": "Keep monitoring regularly, maintain good growing conditions, ensure adequate spacing, and provide consistent care for optimal pepper production."
        },
        "pepper_healthy": {
            "description": "Your pepper plant is healthy with vibrant green leaves and no visible signs of disease! Excellent job maintaining your plant.",
            "treatment": "Continue regular care: water consistently (1-2 inches per week), fertilize every 2-3 weeks, and support heavy fruit branches.",
            "prevention": "Keep monitoring for pests, maintain proper spacing (18-24 inches), ensure good air circulation, and harvest peppers regularly to encourage more fruit production."
        }
    }
    
    # Try exact match first
    if normalized in disease_db:
        print(f"✅ Found exact match: {normalized}")
        return disease_db[normalized]
    
    # Try to find matching disease with partial matching
    for key in disease_db.keys():
        # Check if key contains normalized or vice versa
        if normalized in key or key in normalized:
            print(f"✅ Found partial match: {key}")
            return disease_db[key]
    
    # Try word-by-word matching
    disease_words = set(normalized.split('_'))
    for key, info in disease_db.items():
        key_words = set(key.split('_'))
        # If at least 2 meaningful words match
        common_words = disease_words & key_words
        common_words = {w for w in common_words if len(w) > 3}  # Only meaningful words
        if len(common_words) >= 2:
            print(f"✅ Found word match: {key} (common: {common_words})")
            return info
    
    print(f"⚠️ No match found for: {normalized}")
    
    # Default fallback with plant-specific info
    if 'healthy' in normalized:
        return {
            "description": f"Great news! Your plant appears healthy with no visible signs of disease. The leaves look vibrant and there are no concerning symptoms.",
            "treatment": "Continue your current care routine: consistent watering, proper fertilization, and regular monitoring.",
            "prevention": "Keep up the good work! Monitor regularly, maintain proper spacing, ensure good air circulation, and practice crop rotation."
        }
    
    # Default for diseases
    return {
        "description": f"Disease detected: {disease_name}. This condition requires attention to prevent spread and minimize crop damage.",
        "treatment": "Remove affected leaves immediately, improve air circulation, avoid overhead watering, and consider applying appropriate fungicides or bactericides. Consult local agricultural extension for specific recommendations.",
        "prevention": "Practice good agricultural hygiene: use disease-free seeds, practice crop rotation (3-4 years), maintain proper plant spacing, remove plant debris, and monitor regularly for early detection."
    }

DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "description": "Bacterial spot is caused by Xanthomonas bacteria and appears as dark spots on leaves and fruits.",
        "treatment": "Remove infected leaves, apply copper-based fungicides, avoid overhead watering, ensure good air circulation.",
        "prevention": "Use disease-free seeds, practice crop rotation, maintain proper spacing between plants."
    },
    "Tomato___Early_blight": {
        "description": "Early blight is a fungal disease causing brown spots with concentric rings on older leaves.",
        "treatment": "Remove infected leaves, apply fungicides containing chlorothalonil, improve air circulation.",
        "prevention": "Mulch around plants, water at soil level, practice crop rotation."
    },
    "Tomato___Late_blight": {
        "description": "Late blight is a devastating disease that can destroy entire crops quickly.",
        "treatment": "Remove and destroy infected plants immediately, apply fungicides, avoid overhead watering.",
        "prevention": "Plant resistant varieties, ensure good drainage, monitor weather conditions."
    },
    "Tomato___Leaf_Mold": {
        "description": "Leaf mold is caused by fungus and creates yellow spots on upper leaf surfaces.",
        "treatment": "Improve ventilation, reduce humidity, apply fungicides if severe.",
        "prevention": "Use resistant varieties, maintain proper spacing, avoid overhead irrigation."
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Septoria leaf spot causes small circular spots with dark borders on leaves.",
        "treatment": "Remove infected leaves, apply copper-based fungicides, mulch to prevent splash.",
        "prevention": "Practice crop rotation, use disease-free transplants, maintain proper spacing."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Spider mites are tiny pests that cause yellowing and stippling of leaves.",
        "treatment": "Spray with water to dislodge mites, apply insecticidal soap or neem oil.",
        "prevention": "Maintain adequate moisture, introduce predatory mites, avoid dusty conditions."
    },
    "Tomato___Target_Spot": {
        "description": "Target spot creates brown lesions with concentric rings on leaves and fruits.",
        "treatment": "Remove infected leaves, apply fungicides, improve air circulation.",
        "prevention": "Use resistant varieties, practice crop rotation, avoid overhead watering."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Viral disease transmitted by whiteflies causing leaf curling and yellowing.",
        "treatment": "Remove infected plants, control whitefly population, use reflective mulches.",
        "prevention": "Use virus-free transplants, control whiteflies, use resistant varieties."
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "Mosaic virus causes mottled yellow-green patterns on leaves.",
        "treatment": "Remove and destroy infected plants, disinfect tools, control aphids.",
        "prevention": "Use virus-free seeds, practice good sanitation, control insect vectors."
    },
    "Tomato___healthy": {
        "description": "Plant appears healthy with no visible signs of disease!",
        "treatment": "Continue regular care and monitoring.",
        "prevention": "Maintain good practices: proper watering, fertilization, and pest monitoring."
    },
    "Potato___Early_blight": {
        "description": "Early blight causes dark brown spots with concentric rings on potato leaves.",
        "treatment": "Apply fungicides, remove infected foliage, ensure proper spacing.",
        "prevention": "Use certified seed potatoes, practice crop rotation, maintain soil health."
    },
    "Potato___Late_blight": {
        "description": "Late blight is a serious disease that can destroy potato crops rapidly.",
        "treatment": "Apply protective fungicides, destroy infected plants, harvest early if needed.",
        "prevention": "Plant resistant varieties, monitor weather, ensure good drainage."
    },
    "Potato___healthy": {
        "description": "Potato plant is healthy with no disease symptoms!",
        "treatment": "Continue regular care and monitoring.",
        "prevention": "Maintain good agricultural practices and regular inspection."
    },
    "Pepper,_bell___Bacterial_spot": {
        "description": "Bacterial spot causes dark lesions on pepper leaves and fruits.",
        "treatment": "Apply copper-based bactericides, remove infected leaves, improve air flow.",
        "prevention": "Use disease-free seeds, practice crop rotation, avoid overhead watering."
    },
    "Pepper,_bell___healthy": {
        "description": "Pepper plant is healthy and thriving!",
        "treatment": "Continue regular maintenance.",
        "prevention": "Keep monitoring and maintain good growing conditions."
    }
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path, target_size=(128, 128)):
    """Preprocess image for model prediction"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

def get_disease_info(disease_name):
    """Get disease information and recommendations"""
    return get_disease_info_smart(disease_name)

@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Prepare image for prediction
        img_array = prepare_image(filepath)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx]) * 100
        
        # Get disease name
        disease_name = class_labels[predicted_class_idx]
        
        # Get disease information using smart lookup
        disease_info = get_disease_info(disease_name)
        
        # Clean up - delete uploaded file
        os.remove(filepath)
        
        # Prepare response
        response = {
            'success': True,
            'disease': disease_name.replace('___', ' - ').replace('_', ' '),
            'confidence': round(confidence, 2),
            'description': disease_info['description'],
            'treatment': disease_info['treatment'],
            'prevention': disease_info['prevention'],
            'is_healthy': 'healthy' in disease_name.lower()
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌿 PLANT DISEASE DETECTION WEB APP")
    print("="*70)
    print(f"✅ Model: {MODEL_PATH}")
    print(f"✅ Classes: {len(class_labels)}")
    print("\n🚀 Starting server...")
    print("📱 Open browser: http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)