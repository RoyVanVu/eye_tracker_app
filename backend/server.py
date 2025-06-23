import io 
import torch 
import torch.nn as nn
import base64
import numpy as np 
from PIL import Image 
import cv2
import traceback
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class GazeTrackingModel(nn.Module):
    def __init__(self):
        super(GazeTrackingModel, self).__init__()

        self.left_eye_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.right_eye_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        sample_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            cnn_output_size = self._get_cnn_output_size(sample_input)

        self.fc_combined = nn.Sequential(
            nn.Linear(cnn_output_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )
    
    def _get_cnn_output_size(self, x):
        with torch.no_grad():
            x = self.left_eye_cnn(x)
            return x.view(1, -1).shape[1]
    
    def forward(self, eye_left, eye_right):
        left_eye_features = self.left_eye_cnn(eye_left)
        right_eye_features = self.right_eye_cnn(eye_right)
        combined_features = torch.cat([
            left_eye_features,
            right_eye_features,
        ], dim=1)
        gaze_output = self.fc_combined(combined_features)

        return gaze_output

def preprocess_image(base64_image):
    try:
        if ',' in base64_image:
            image_data = base64.b64decode(base64_image.split(',')[1])
        else:
            image_data = base64.b64decode(base64_image)
        
        image = Image.open(io.BytesIO(image_data))
        logger.info(f"Image loaded: size={image.size}, mode={image.mode}")
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_np = np.array(image)
        image_resized = cv2.resize(image_np, (224, 224))

        tensor = torch.from_numpy(image_resized).float()
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0)

        return tensor
    except Exception as e:
        logger.error(f"Error in preprocess_image: {e}")
        logger.error(traceback.format_exc())
        return None

print("Loading eye tracking model...")
model = GazeTrackingModel()
model_path = "/app/model/gaze_model_epoch70_normalize28(eye image, gaze coordinate)_CosineAnnealingLR(20, 1e-7)_val_loss_0.0809_normalize1Dataset_batch32"
try:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure the model path is correct and the file exists")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

prediction_counter = 0
calibration_data = []
calibrated_model = None
max_calibration_samples = 50
is_calibration_active = False

@app.route('/predict', methods=['POST'])
def predict():
    global prediction_counter
    prediction_counter += 1

    try:
        logger.info(f"\n=== Prediction request #{prediction_counter} ===")
        data = request.json 
        if not data:
            logger.error("No JSON data received")
            return jsonify({
                'error': 'No JSON data received'
            }), 400
        
        left_eye_base64 = data.get('leftEye')
        right_eye_base64 = data.get('rightEye')
        if not left_eye_base64 or not right_eye_base64:
            logger.error("Missing eye image data")
            return jsonify({
                'error': 'Both left and right eye images are required'
            }), 400
        
        left_eye_tensor = preprocess_image(left_eye_base64)
        if left_eye_tensor is None:
            return jsonify({
                'error': 'Failed to process left eye image'
            }), 400
        left_eye_tensor = left_eye_tensor.to(device)

        right_eye_tensor = preprocess_image(right_eye_base64)
        if right_eye_tensor is None:
            return jsonify({
                'error': 'Failed to process right eye image'
            }), 400
        right_eye_tensor = right_eye_tensor.to(device)

        logger.info("Running model inference...")
        with torch.no_grad():
            prediction = model(left_eye_tensor, right_eye_tensor)
        gaze_coords = prediction[0].cpu().numpy().tolist()
        logger.info(f"Final gaze coordinates: x={gaze_coords[0]:.4f}, y={gaze_coords[1]:.4f}")
        
        response_data = {
            'gaze': {
                'x': float(gaze_coords[0]),
                'y': float(gaze_coords[1])
            },
            'prediction_id': prediction_counter
        }
        logger.info(f"Sending response: {response_data}")
        return jsonify(response_data)
    except Exception as e:
        logger.info(f"Error in prediction: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    global calibrated_model, calibration_data, is_calibration_active

    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'model_loaded': True,
        'predictions_made': prediction_counter,
        'calibration': {
            'is_active': is_calibration_active,
            'samples_collected': len(calibration_data),
            'has_calibrated_model': calibrated_model is not None,
            'active_model': 'calibrated' if calibrated_model is not None else 'original'
        }
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    logger.info("Test endpoint called")
    return jsonify({
        'message': 'Server is working!',
        'device': str(device),
        'predictions_made': prediction_counter
    })

@app.route('/calibration/start', methods=['POST'])
def start_calibration():
    global calibration_data, is_calibration_active

    try:
        calibration_data = []
        is_calibration_active = True
        logger.info("Calibration session started")
        return jsonify({
            'status': 'success',
            'message': 'Calibration started',
            'max_samples': max_calibration_samples,
            'current_samples': 0
        })
    except Exception as e:
        logger.error(f"Error starting calibration: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/calibration/status', methods=['GET'])
def calibration_status():
    global calibration_data, is_calibration_active, calibrated_model

    return jsonify({
        'is_active': is_calibration_active,
        'samples_collected': len(calibration_data),
        'max_samples': max_calibration_samples,
        'progress': len(calibration_data) / max_calibration_samples if max_calibration_samples > 0 else 0,
        'can_finish': len(calibration_data) >= 10,
        'has_calibrated_model': calibrated_model is not None
    })

@app.route('/calibration/reset', methods=['POST'])
def reset_calibration():
    global calibration_data, is_calibration_active, calibrated_model
    calibration_data = []
    is_calibration_active = False
    calibrated_model = None
    logger.info("Calibration reset")

    return jsonify({
        'status': 'success', 'message': 'Calibration reset'
    })

@app.route('/calibration/test', methods=['GET'])
def test_calibration():
    """Test endpoint to verify calibration functionality"""
    return jsonify({
        'message': 'Calibration endpoints are working!',
        'available_endpoints': [
            '/calibration/start',
            '/calibration/add_sample',
            '/calibration/status',
            '/calibration/reset',
            '/calibration/test'
        ],
        'current_status': {
            'is_active': is_calibration_active,
            'samples_collected': len(calibration_data),
            'has_calibrated_model': calibrated_model is not None
        }
    })

@app.route('/calibration/add_sample', methods=['POST'])
def add_calibration_sample():
    """Add a calibration sample (eye images + target gaze position)"""
    global calibration_data, is_calibration_active
    if not is_calibration_active:
        return jsonify({
            'error': 'Calibration not active. Please start calibration first.'
        }), 400
    
    try:
        data = request.json
        if not data:
            return jsonify({
                'error': 'No JSON data received'
            }), 400

        required_fields = ['leftEye', 'rightEye', 'targetX', 'targetY']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        left_eye_tensor = preprocess_image(data['leftEye'])
        right_eye_tensor = preprocess_image(data['rightEye'])
        if left_eye_tensor is None or right_eye_tensor is None:
            return jsonify({
                'error': 'Failed to process eye images'
            }), 400
        
        target_x = float(data['targetX'])
        target_y = float(data['targetY'])
        if not (0 <= target_x <= 1 and 0 <= target_y <= 1):
            return jsonify({
                'error': 'Target coordinates must be between 0 and 1'
            }), 400

        sample = {
            'left_eye': left_eye_tensor.cpu(),
            'right_eye': right_eye_tensor.cpu(),
            'target_gaze': torch.tensor([target_x, target_y], dtype=torch.float32)
        }
        calibration_data.append(sample)

        current_count = len(calibration_data)
        logger.info(f"Added calibration sample {current_count}/{max_calibration_samples}")
        logger.info(f"Target: ({target_x:.3f}, {target_y:.3f})")

        if current_count >= max_calibration_samples:
            logger.info(f"Maximum calibration samples ({max_calibration_samples}) reached!")
        
        return jsonify({
            'status': 'success',
            'message': 'Sample added successfully',
            'samples_collected': current_count,
            'max_samples': max_calibration_samples,
            'progress': current_count / max_calibration_samples,
            'can_finish': current_count >= 10,
            'is_complete': current_count >= max_calibration_samples
        })
    
    except ValueError as e:
        logger.error(f"Invalid data format: {e}")
        return jsonify({
            'error': f'Invalid data format: {str(e)}'
        }), 400
    except Exception as e:
        logger.error(f"Error adding calibration sample: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/calibration/add_test_sample', methods=['POST'])
def add_test_calibration_sample():
    global calibration_data, is_calibration_active
    if not is_calibration_active:
        return jsonify({
            'error': 'Calibration is not active. Please start calibration first.'
        }), 400
    
    try:
        data = request.json or {}

        target_x = float(data.get('targetX', 0.5))
        target_y = float(data.get('targetY', 0.5))
        if not (0 <= target_x <= 1 and 0 <= target_y <= 1):
            return jsonify({
                'error': 'Target coordinates must be between 0 and 1'
            }), 400
        
        dummy_left = torch.randn(1, 3, 224, 224)
        dummy_right = torch.randn(1, 3, 224, 224)
        sample = {
            'left_eye': dummy_left.cpu(),
            'right_eye': dummy_right.cpu(),
            'target_gaze': torch.tensor([target_x, target_y], dtype=torch.float32)
        }
        calibration_data.append(sample)

        current_count = len(calibration_data)
        logger.info(f"Added test calibration sample {current_count}/{max_calibration_samples}")
        logger.info(f"Target: ({target_x:.3f}, {target_y:.3f})")

        return jsonify({
            'status': 'success',
            'message': 'Test sample add successfully',
            'samples_collected': current_count,
            'max_samples': max_calibration_samples,
            'progress': current_count / max_calibration_samples,
            'can_finish': current_count >= 10,
            'is_complete': current_count >= max_calibration_samples
        })
    
    except ValueError as e:
        logger.error(f"Invalid data format: {e}")
        return jsonify({
            'error': f'Invalid data format: {str(e)}'
        }), 400
    except Exception as e:
        logger.error(f"Error adding test calibration sample: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Internal server error {str(e)}'
        }), 500

if __name__ == '__main__':
    logger.info('Starting Flask server...')
    app.run(host='0.0.0.0', port=5000, debug=True)