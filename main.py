from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from enum import Enum
import os
import uuid
from datetime import datetime
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download mode configuration
class DownloadMode(str, Enum):
    ON = "on"           # Always save all images
    OFF = "off"         # Never save images
    AUTO = "auto"       # Save only confident predictions
    CONFIDENT = "confident"  # Save only high confidence (similar to auto)

class DownloadConfig(BaseModel):
    enabled: bool
    mode: DownloadMode
    breed_threshold: float = 0.4    # Minimum confidence for breed predictions
    unknown_threshold: float = 0.5  # Minimum confidence for unknown predictions

# Global download configuration
DOWNLOAD_CONFIG = DownloadConfig(
    enabled=False,
    mode=DownloadMode.OFF,
    breed_threshold=0.4,
    unknown_threshold=0.5
)

app = FastAPI(
    title="Cattle Breed Classification API with Download Control",
    description="AI-powered cattle breed identification with configurable download/save options",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = 'corrected_cow_breed_model.h5'
CORRECTED_MODEL_PATH = 'best_corrected_cow_breed_model.h5'
LABELS_PATH = 'class_labels.txt'
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
IMG_SIZE = (224, 224)
DATASET_BASE_PATH = "dataset/original"

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

def get_next_test_number(breed_folder):
    """Get the next available test number for the breed folder"""
    if not os.path.exists(breed_folder):
        return 1
    
    existing_files = [f for f in os.listdir(breed_folder) if f.startswith('test_')]
    if not existing_files:
        return 1
    
    # Extract numbers from existing test files
    numbers = []
    for file in existing_files:
        try:
            # Extract number between 'test_' and file extension
            parts = file.split('_')
            if len(parts) >= 2:
                number_part = parts[-1].split('.')[0]
                if number_part.isdigit():
                    numbers.append(int(number_part))
                else:
                    # Handle timestamp format: test_breed_N_timestamp.ext
                    for part in reversed(parts):
                        if part.split('.')[0].isdigit():
                            numbers.append(int(part.split('.')[0]))
                            break
        except (ValueError, IndexError):
            continue
    
    return max(numbers) + 1 if numbers else 1

def should_save_image(predicted_breed: str, confidence: float, unknown_confidence: float, 
                     classification_type: str) -> tuple[bool, str]:
    """
    Determine if image should be saved based on download configuration
    Returns (should_save, reason)
    """
    if not DOWNLOAD_CONFIG.enabled:
        return False, "Download disabled globally"
    
    mode = DOWNLOAD_CONFIG.mode
    
    if mode == DownloadMode.OFF:
        return False, "Download mode set to OFF"
    
    elif mode == DownloadMode.ON:
        return True, "Download mode set to ON (save all)"
    
    elif mode in [DownloadMode.AUTO, DownloadMode.CONFIDENT]:
        # Check confidence thresholds
        if predicted_breed.lower() != 'unknown' and confidence >= DOWNLOAD_CONFIG.breed_threshold:
            return True, f"Breed confidence {confidence*100:.1f}% ≥ {DOWNLOAD_CONFIG.breed_threshold*100}% threshold"
        
        elif predicted_breed.lower() == 'unknown' and unknown_confidence >= DOWNLOAD_CONFIG.unknown_threshold:
            return True, f"Unknown confidence {unknown_confidence*100:.1f}% ≥ {DOWNLOAD_CONFIG.unknown_threshold*100}% threshold"
        
        else:
            return False, f"Below confidence thresholds (breed: {confidence*100:.1f}%, unknown: {unknown_confidence*100:.1f}%)"
    
    return False, "Unknown download mode"

def save_uploaded_image(temp_file_path, predicted_breed, confidence, unknown_confidence, 
                       classification_type, original_filename):
    """Save the uploaded image to appropriate dataset folder based on download config"""
    
    # Check if we should save this image
    should_save, reason = should_save_image(predicted_breed, confidence, unknown_confidence, classification_type)
    
    if not should_save:
        logger.info(f"Image not saved: {reason}")
        return {
            "saved": False,
            "reason": reason,
            "download_config": {
                "enabled": DOWNLOAD_CONFIG.enabled,
                "mode": DOWNLOAD_CONFIG.mode.value,
                "breed_threshold": DOWNLOAD_CONFIG.breed_threshold,
                "unknown_threshold": DOWNLOAD_CONFIG.unknown_threshold
            }
        }
        
    try:
        # Create dataset base directory if it doesn't exist
        os.makedirs(DATASET_BASE_PATH, exist_ok=True)
        
        # Determine the breed folder name
        if classification_type == 'low_confidence':
            breed_folder_name = 'low_confidence'
        elif predicted_breed.lower() == 'unknown' or classification_type in ['outlier', 'unknown_confident']:
            breed_folder_name = 'unknown'
        else:
            breed_folder_name = predicted_breed.lower().replace(' ', '_')
        
        # Create the target folder
        target_folder = os.path.join(DATASET_BASE_PATH, breed_folder_name)
        os.makedirs(target_folder, exist_ok=True)
        
        # Get next test number
        test_number = get_next_test_number(target_folder)
        
        # Get original file extension
        original_extension = Path(original_filename).suffix.lower()
        if not original_extension:
            original_extension = '.jpg'  # Default extension
        
        # Create new filename with timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"test_{breed_folder_name}_{test_number}_{timestamp}{original_extension}"
        target_path = os.path.join(target_folder, new_filename)
        
        # Copy the image
        shutil.copy2(temp_file_path, target_path)
        
        logger.info(f"Image saved to: {target_path}")
        return {
            "saved": True,
            "saved_path": target_path,
            "saved_filename": new_filename,
            "saved_folder": breed_folder_name,
            "reason": reason,
            "download_config": {
                "enabled": DOWNLOAD_CONFIG.enabled,
                "mode": DOWNLOAD_CONFIG.mode.value,
                "breed_threshold": DOWNLOAD_CONFIG.breed_threshold,
                "unknown_threshold": DOWNLOAD_CONFIG.unknown_threshold
            }
        }
        
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return {
            "saved": False,
            "error": str(e),
            "reason": f"Save failed: {e}",
            "download_config": {
                "enabled": DOWNLOAD_CONFIG.enabled,
                "mode": DOWNLOAD_CONFIG.mode.value,
                "breed_threshold": DOWNLOAD_CONFIG.breed_threshold,
                "unknown_threshold": DOWNLOAD_CONFIG.unknown_threshold
            }
        }

def load_class_labels(labels_file=LABELS_PATH):
    """Load class labels from file"""
    if not os.path.exists(labels_file):
        logger.error(f"Labels file not found: {labels_file}")
        return None
    
    try:
        with open(labels_file, 'r') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
        return labels
    except Exception as e:
        logger.error(f"Error loading labels: {e}")
        return None

def preprocess_image(img_path):
    """Preprocess image for prediction"""
    try:
        img = Image.open(img_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss function for loading corrected model"""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        ce = -y_true * tf.math.log(y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        focal_loss = focal_weight * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))
    
    return focal_loss_fixed

def predict_breed_builtin(model_path, img_path, labels_file=LABELS_PATH, top_k=5):
    """Built-in prediction function with enhanced classification logic"""
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return None
        
        logger.info(f"Loading model: {model_path}")
        
        # Try loading with focal loss first
        try:
            model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()})
            logger.info("Loaded model with focal loss")
        except Exception as e:
            logger.warning(f"Failed to load with focal loss, trying standard: {e}")
            try:
                model = load_model(model_path)
                logger.info("Loaded model with standard method")
            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                return None
        
        # Load class labels
        class_labels = load_class_labels(labels_file)
        if class_labels is None:
            return None
        
        # Preprocess image
        img_array = preprocess_image(img_path)
        if img_array is None:
            return None
        
        # Make prediction
        logger.info("Making prediction...")
        predictions = model.predict(img_array, verbose=0)
        
        # Convert predictions to regular Python types immediately
        pred_array = predictions[0].astype(float)  # Convert to Python float array
        
        # Enhanced classification logic
        max_confidence = float(np.max(pred_array))
        predicted_idx = int(np.argmax(pred_array))
        predicted_class = class_labels[predicted_idx]
        
        # Check for unknown class
        has_unknown = 'unknown' in class_labels
        unknown_confidence = 0.0
        unknown_idx = -1
        if has_unknown:
            unknown_idx = class_labels.index('unknown')
            unknown_confidence = float(pred_array[unknown_idx])
        
        # Classification decision logic
        classification_type = 'breed_confident'
        is_outlier = False
        
        if predicted_class.lower() != 'unknown' and max_confidence >= DOWNLOAD_CONFIG.breed_threshold:
            classification_type = 'breed_confident'
            is_outlier = False
        elif has_unknown and predicted_class.lower() == 'unknown' and unknown_confidence >= DOWNLOAD_CONFIG.unknown_threshold:
            classification_type = 'unknown_confident'
            is_outlier = True  # Treat confident unknown as outlier for API response
        elif has_unknown and unknown_confidence >= DOWNLOAD_CONFIG.unknown_threshold:
            classification_type = 'unknown_confident'
            is_outlier = True
        else:
            classification_type = 'low_confidence'
            is_outlier = True
        
        # Get top predictions
        top_indices = np.argsort(pred_array)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            breed = class_labels[int(idx)]
            confidence = float(pred_array[int(idx)])
            results.append((breed, confidence))
        
        # Calculate entropy
        epsilon = 1e-10
        entropy_calc = pred_array * np.log(pred_array + epsilon)
        entropy = float(-np.sum(entropy_calc))
        max_entropy = float(np.log(len(pred_array)))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Decision reason
        if classification_type == 'breed_confident':
            decision_reason = f"High breed confidence: {max_confidence*100:.1f}% ≥ {DOWNLOAD_CONFIG.breed_threshold*100}%"
        elif classification_type == 'unknown_confident':
            decision_reason = f"High unknown confidence: {unknown_confidence*100:.1f}% ≥ {DOWNLOAD_CONFIG.unknown_threshold*100}%"
        else:
            decision_reason = f"Low confidence - breed: {max_confidence*100:.1f}%, unknown: {unknown_confidence*100:.1f}%"
        
        # Return everything as Python native types
        return {
            'results': results,
            'predicted_class': predicted_class,
            'classification_type': classification_type,
            'is_outlier': is_outlier,
            'reason': decision_reason,
            'max_confidence': max_confidence,
            'unknown_confidence': unknown_confidence,
            'has_unknown_class': has_unknown,
            'entropy': normalized_entropy
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None

def create_temp_file(file_content: bytes, original_filename: str) -> str:
    """Create temporary file"""
    file_ext = os.path.splitext(original_filename)[1]
    temp_filename = f"temp_{uuid.uuid4()}{file_ext}"
    
    with open(temp_filename, "wb") as f:
        f.write(file_content)
    
    return temp_filename

def format_breed_name(breed_name: str) -> str:
    """Format breed name for React frontend"""
    if breed_name.lower() == 'unknown':
        return 'unknown'
    
    formatted = breed_name.replace('_', ' ').lower()
    
    breed_mapping = {
        'gir': 'gir',
        'sahiwal': 'sahiwal', 
        'kankrej': 'kankrej',
        'murrah': 'murrah',
        'jaffarabadi': 'jaffarabadi',
        'sahiwal cross': 'sahiwal_cross',
        'fresian': 'fresian',
        'fresian cross': 'fresian_cross',
        'brahman': 'brahman',
        'brahman cross': 'brahman_cross'
    }
    
    return breed_mapping.get(formatted, formatted)

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint with download config info"""
    corrected_available = os.path.exists(CORRECTED_MODEL_PATH)
    standard_available = os.path.exists(MODEL_PATH)
    labels_available = os.path.exists(LABELS_PATH)
    
    return {
        "message": "Cattle Breed Classification API v4.0 - Download Control",
        "status": "healthy",
        "models": {
            "corrected_model": corrected_available,
            "standard_model": standard_available,
            "labels_file": labels_available
        },
        "active_model": CORRECTED_MODEL_PATH if corrected_available else MODEL_PATH,
        "download_config": {
            "enabled": DOWNLOAD_CONFIG.enabled,
            "mode": DOWNLOAD_CONFIG.mode.value,
            "breed_threshold": DOWNLOAD_CONFIG.breed_threshold,
            "unknown_threshold": DOWNLOAD_CONFIG.unknown_threshold,
            "save_path": DATASET_BASE_PATH
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check with download status"""
    corrected_available = os.path.exists(CORRECTED_MODEL_PATH)
    standard_available = os.path.exists(MODEL_PATH)
    labels_available = os.path.exists(LABELS_PATH)
    
    if not (corrected_available or standard_available):
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "No model files found",
                "details": f"Looking for: {CORRECTED_MODEL_PATH} or {MODEL_PATH}"
            }
        )
    
    if not labels_available:
        return JSONResponse(
            status_code=503, 
            content={
                "status": "unhealthy",
                "error": "Labels file not found",
                "details": f"Looking for: {LABELS_PATH}"
            }
        )
    
    return {
        "status": "healthy",
        "models": {
            "corrected_model": corrected_available,
            "standard_model": standard_available
        },
        "active_model": CORRECTED_MODEL_PATH if corrected_available else MODEL_PATH,
        "labels_available": labels_available,
        "download_config": {
            "enabled": DOWNLOAD_CONFIG.enabled,
            "mode": DOWNLOAD_CONFIG.mode.value,
            "breed_threshold": DOWNLOAD_CONFIG.breed_threshold,
            "unknown_threshold": DOWNLOAD_CONFIG.unknown_threshold,
            "save_path": DATASET_BASE_PATH
        }
    }

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    download_override: Optional[DownloadMode] = Query(None, description="Override download mode for this prediction")
):
    """Predict cattle breed with configurable download control"""
    
    # Temporarily override download mode if requested
    original_mode = DOWNLOAD_CONFIG.mode
    if download_override:
        DOWNLOAD_CONFIG.mode = download_override
        logger.info(f"Download mode temporarily overridden to: {download_override}")
    
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        try:
            content = await file.read()
            file_size = len(content)
            
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB"
                )
                
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            raise HTTPException(status_code=400, detail="Error reading uploaded file")
        
        temp_file = None
        try:
            # Create temporary file
            temp_file = create_temp_file(content, file.filename)
            logger.info(f"Processing: {file.filename} ({file_size} bytes)")
            
            # Choose model
            model_path = CORRECTED_MODEL_PATH if os.path.exists(CORRECTED_MODEL_PATH) else MODEL_PATH
            
            if not os.path.exists(model_path):
                raise HTTPException(status_code=503, detail="No model file available")
            
            # Make prediction
            prediction_result = predict_breed_builtin(model_path, temp_file, LABELS_PATH, top_k=5)
            
            if not prediction_result:
                raise HTTPException(status_code=500, detail="Prediction failed")
            
            # Convert all results to ensure JSON serialization
            prediction_result = convert_numpy_types(prediction_result)
            
            # Save the uploaded image based on download configuration
            save_info = save_uploaded_image(
                temp_file,
                prediction_result['predicted_class'],
                prediction_result['max_confidence'],
                prediction_result['unknown_confidence'],
                prediction_result['classification_type'],
                file.filename
            )
            
            # Handle outlier detection
            if prediction_result['is_outlier']:
                logger.info(f"Outlier/Unknown detected: {prediction_result['reason']}")
                
                response_data = {
                    "success": False,
                    "predictions": [],
                    "is_known_breed": False,
                    "error": "outlier_detected",
                    "message": "This image appears to be an unknown breed or not a cow/buffalo",
                    "details": {
                        "classification_type": prediction_result['classification_type'],
                        "reason": prediction_result['reason'],
                        "max_confidence": prediction_result['max_confidence'],
                        "unknown_confidence": prediction_result['unknown_confidence'],
                        "entropy": prediction_result['entropy']
                    },
                    "suggestion": "Please upload a clear image of a supported cattle breed",
                    "download_info": save_info
                }
                
                return convert_numpy_types(response_data)
            
            # Format successful predictions
            predictions = []
            for i, (breed_name, confidence) in enumerate(prediction_result['results'][:5]):
                if breed_name.lower() == 'unknown':
                    # Only include unknown if it's very confident
                    if confidence > 0.8:
                        predictions.append({
                            "rank": len(predictions) + 1,
                            "breed_name": "Unknown/Unrecognized Breed",
                            "confidence": confidence,
                            "confidence_percentage": f"{confidence * 100:.2f}%"
                        })
                    continue
                
                predictions.append({
                    "rank": len(predictions) + 1,
                    "breed_name": format_breed_name(breed_name),
                    "confidence": confidence,
                    "confidence_percentage": f"{confidence * 100:.2f}%"
                })
            
            if not predictions:
                return convert_numpy_types({
                    "success": False,
                    "predictions": [],
                    "error": "no_valid_predictions",
                    "message": "No valid breed predictions available",
                    "download_info": save_info
                })
            
            success_response = {
                "success": True,
                "predictions": predictions,
                "is_known_breed": True,
                "confidence_score": prediction_result['max_confidence'],
                "classification_details": {
                    "type": prediction_result['classification_type'],
                    "reason": prediction_result['reason'],
                    "meets_breed_threshold": prediction_result['max_confidence'] >= DOWNLOAD_CONFIG.breed_threshold,
                    "meets_unknown_threshold": prediction_result['unknown_confidence'] >= DOWNLOAD_CONFIG.unknown_threshold
                },
                "model_info": {
                    "type": "corrected" if "corrected" in model_path else "standard",
                    "has_unknown_class": prediction_result['has_unknown_class'],
                    "breed_prioritized": True
                },
                "download_info": save_info
            }
            
            return convert_numpy_types(success_response)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            error_response = {
                "success": False,
                "predictions": [],
                "error": "prediction_failed",
                "message": f"Prediction failed: {str(e)}",
                "download_info": {"saved": False, "reason": "Prediction failed"}
            }
            return convert_numpy_types(error_response)
            
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file: {e}")
    
    finally:
        # Restore original download mode if it was overridden
        if download_override:
            DOWNLOAD_CONFIG.mode = original_mode

@app.get("/download/config")
async def get_download_config():
    """Get current download configuration"""
    return {
        "success": True,
        "config": {
            "enabled": DOWNLOAD_CONFIG.enabled,
            "mode": DOWNLOAD_CONFIG.mode.value,
            "breed_threshold": DOWNLOAD_CONFIG.breed_threshold,
            "unknown_threshold": DOWNLOAD_CONFIG.unknown_threshold,
            "save_path": DATASET_BASE_PATH
        },
        "available_modes": [mode.value for mode in DownloadMode],
        "mode_descriptions": {
            "on": "Always save all images to appropriate folders",
            "off": "Never save images (prediction only)",
            "auto": "Save only confident predictions (breed ≥ threshold OR unknown ≥ threshold)",
            "confident": "Same as auto - save only high confidence predictions"
        }
    }

@app.post("/download/config")
async def update_download_config(
    enabled: Optional[bool] = None,
    mode: Optional[DownloadMode] = None,
    breed_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    unknown_threshold: Optional[float] = Query(None, ge=0.0, le=1.0)
):
    """Update download configuration"""
    global DOWNLOAD_CONFIG
    
    changes = []
    
    if enabled is not None:
        old_enabled = DOWNLOAD_CONFIG.enabled
        DOWNLOAD_CONFIG.enabled = enabled
        changes.append(f"enabled: {old_enabled} → {enabled}")
    
    if mode is not None:
        old_mode = DOWNLOAD_CONFIG.mode
        DOWNLOAD_CONFIG.mode = mode
        changes.append(f"mode: {old_mode.value} → {mode.value}")
    
    if breed_threshold is not None:
        old_threshold = DOWNLOAD_CONFIG.breed_threshold
        DOWNLOAD_CONFIG.breed_threshold = breed_threshold
        changes.append(f"breed_threshold: {old_threshold} → {breed_threshold}")
    
    if unknown_threshold is not None:
        old_threshold = DOWNLOAD_CONFIG.unknown_threshold
        DOWNLOAD_CONFIG.unknown_threshold = unknown_threshold
        changes.append(f"unknown_threshold: {old_threshold} → {unknown_threshold}")
    
    logger.info(f"Download config updated: {', '.join(changes) if changes else 'no changes'}")
    
    return {
        "success": True,
        "message": f"Download configuration updated: {', '.join(changes) if changes else 'no changes'}",
        "config": {
            "enabled": DOWNLOAD_CONFIG.enabled,
            "mode": DOWNLOAD_CONFIG.mode.value,
            "breed_threshold": DOWNLOAD_CONFIG.breed_threshold,
            "unknown_threshold": DOWNLOAD_CONFIG.unknown_threshold,
            "save_path": DATASET_BASE_PATH
        },
        "changes": changes
    }

@app.get("/breeds")
async def get_supported_breeds():
    """Get list of supported breeds"""
    try:
        class_labels = load_class_labels()
        if not class_labels:
            raise HTTPException(status_code=503, detail="Could not load breed list")
        
        breeds = []
        for breed in class_labels:
            if breed.lower() != 'unknown':
                breeds.append({
                    "name": breed.replace('_', ' ').title(),
                    "key": format_breed_name(breed)
                })
        
        return {
            "success": True,
            "breeds": breeds,
            "total_breeds": len(breeds),
            "has_outlier_detection": 'unknown' in class_labels,
            "download_config": {
                "enabled": DOWNLOAD_CONFIG.enabled,
                "mode": DOWNLOAD_CONFIG.mode.value
            }
        }
        
    except Exception as e:
        logger.error(f"Error loading breeds: {e}")
        raise HTTPException(status_code=500, detail="Error loading breed list")

@app.get("/dataset/stats")
async def get_dataset_stats():
    """Get statistics about saved images in the dataset"""
    try:
        if not os.path.exists(DATASET_BASE_PATH):
            return {
                "success": True,
                "stats": {
                    "total_saved_images": 0,
                    "breeds": {},
                    "download_config": {
                        "enabled": DOWNLOAD_CONFIG.enabled,
                        "mode": DOWNLOAD_CONFIG.mode.value,
                        "breed_threshold": DOWNLOAD_CONFIG.breed_threshold,
                        "unknown_threshold": DOWNLOAD_CONFIG.unknown_threshold
                    },
                    "dataset_path": DATASET_BASE_PATH
                }
            }
        
        breed_stats = {}
        total_images = 0
        
        for breed_folder in os.listdir(DATASET_BASE_PATH):
            breed_path = os.path.join(DATASET_BASE_PATH, breed_folder)
            if os.path.isdir(breed_path):
                test_images = [f for f in os.listdir(breed_path) if f.startswith('test_')]
                image_count = len(test_images)
                breed_stats[breed_folder] = {
                    "count": image_count,
                    "latest_test_number": get_next_test_number(breed_path) - 1
                }
                total_images += image_count
        
        return {
            "success": True,
            "stats": {
                "total_saved_images": total_images,
                "breeds": breed_stats,
                "download_config": {
                    "enabled": DOWNLOAD_CONFIG.enabled,
                    "mode": DOWNLOAD_CONFIG.mode.value,
                    "breed_threshold": DOWNLOAD_CONFIG.breed_threshold,
                    "unknown_threshold": DOWNLOAD_CONFIG.unknown_threshold
                },
                "dataset_path": DATASET_BASE_PATH
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting dataset stats: {e}")
        raise HTTPException(status_code=500, detail="Error getting dataset statistics")

if __name__ == "__main__":
    import uvicorn
    
    print("🐄 Cattle Breed Classification API v4.0 - Download Control")
    print("="*70)
    
    if os.path.exists(CORRECTED_MODEL_PATH):
        print(f"✅ Corrected model found: {CORRECTED_MODEL_PATH}")
    elif os.path.exists(MODEL_PATH):  
        print(f"⚠️  Using standard model: {MODEL_PATH}")
    else:
        print("❌ No model found!")
    
    if os.path.exists(LABELS_PATH):
        print(f"✅ Labels found: {LABELS_PATH}")
    else:
        print(f"❌ Labels not found: {LABELS_PATH}")
    
    print(f"\n📥 Download Configuration:")
    print(f"   • Enabled: {'✅ YES' if DOWNLOAD_CONFIG.enabled else '❌ NO'}")
    print(f"   • Mode: {DOWNLOAD_CONFIG.mode.value.upper()}")
    print(f"   • Breed threshold: {DOWNLOAD_CONFIG.breed_threshold*100}%")
    print(f"   • Unknown threshold: {DOWNLOAD_CONFIG.unknown_threshold*100}%")
    
    mode_explanations = {
        DownloadMode.ON: "All images saved to appropriate folders",
        DownloadMode.OFF: "No images saved (prediction only)",
        DownloadMode.AUTO: f"Only confident predictions saved (breed ≥{DOWNLOAD_CONFIG.breed_threshold*100}% OR unknown ≥{DOWNLOAD_CONFIG.unknown_threshold*100}%)",
        DownloadMode.CONFIDENT: "Same as AUTO mode"
    }
    print(f"   • Behavior: {mode_explanations[DOWNLOAD_CONFIG.mode]}")
    
    if DOWNLOAD_CONFIG.enabled and DOWNLOAD_CONFIG.mode != DownloadMode.OFF:
        print(f"   • Save path: {DATASET_BASE_PATH}/{{breed}}/test_{{breed}}_{{n}}_{{timestamp}}.jpg")
        os.makedirs(DATASET_BASE_PATH, exist_ok=True)
    
    print(f"\n🔧 API Features:")
    print(f"   • GET /download/config - View current download settings")
    print(f"   • POST /download/config - Update download settings")
    print(f"   • POST /predict?download_override=MODE - Override download mode per request")
    print(f"   • Available modes: on, off, auto, confident")
    
    print(f"\n🌐 Server Info:")
    print(f"   • Starting on http://0.0.0.0:8000")
    print(f"   • Mobile access: http://YOUR_IP:8000")
    print(f"   • API docs: http://YOUR_IP:8000/docs")
    print("="*70)
    
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )