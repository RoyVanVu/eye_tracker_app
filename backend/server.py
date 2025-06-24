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

def gaze_loss(predicted, actual, beta=1.0):
    smooth_l1 = F.smooth_l1_loss(predicted, actual)
    euclidean_distance = torch.sqrt(torch.sum((predicted - actual) ** 2, dim=1)).mean()
    return smooth_l1 + beta * euclidean_distance

def train_calibrated_model(calibration_samples, base_model, device, epochs=20, lr=1e-6, beta=0.5):
    logger.info(f"Starting calibration training with {len(calibration_samples)} samples...")

    calibrated_model = copy.deepcopy(base_model)
    calibrated_model.train()
    left_eyes = []
    right_eyes = []
    targets = []
    for sample in calibration_samples:
        left_eyes.append(sample['left_eye'])
        right_eyes.append(sample['right_eye'])
        targets.append(sample['target_gaze'])
    
    left_batch = torch.stack(left_eyes).to(device)
    right_batch = torch.stack(right_eyes).to(device)
    target_batch = torch.stack(targets).to(device)

    logger.info(f"Training data shapes:")
    logger.info(f"Left eyes: {left_batch.shape}")
    logger.info(f"Right eyes: {right_batch.shape}")
    logger.info(f"Targets: {target_batch.shape}")

    optimizer = optim.AdamW(
        calibrated_model.parameters(),
        lr=lr,
        weight_decay=0.0001,
        amsgrad=False
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_loss = float('inf')
    training_losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        predictions = calibrated_model(left_batch, right_batch)
        loss = gaze_loss(predictions, target_batch, beta=beta)

        loss.backward()
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()
        training_losses.append(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else lr
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {current_loss:.6f}, LR = {scheduler.get_last_lr()[0]:2e}")
    
    calibrated_model.eval()
    with torch.no_grad():
        final_predictions = calibrated_model(left_batch, right_batch)
        final_loss = gaze_loss(final_predictions, target_batch, beta=beta).item()
        mse_loss = F.mse_loss(final_predictions, target_batch).item()
        pixel_errors = torch.abs(final_predictions - target_batch) * torch.tensor([1920, 1000]).to(device)
        avg_pixel_error = pixel_errors.mean().item()
        euclidean_errors = torch.sqrt(torch.sum((final_predictions - target_batch) ** 2, dim=1))
        avg_euclidean_error = euclidean_errors.mean().item()
    
    logger.info(f"Calibration training completed!")
    logger.info(f"Final loss: {final_loss:.6f}")
    logger.info(f"Best loss: {best_loss:.6f}")
    logger.info(f"Average pixel error: {avg_pixel_error:.1f} pixels")

    return {
        'model': calibrated_model,
        'final_loss': final_loss,
        'final_mse_loss': mse_loss,
        'best_loss': best_loss,
        'training_losses': training_losses,
        'avg_pixel_error': avg_pixel_error,
        'avg_euclidean_error': avg_euclidean_error,
        'epochs_trained': epochs,
        'samples_used': len(calibration_samples)
    }

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

        active_model_name = 'calibrated' if calibrated_model is not None else 'original'
        logger.info(f"Using {active_model_name} model for prediction")

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
            'prediction_id': prediction_counter,
            'model_info': {
                'active_model': active_model_name,
                'is_calibrated': calibrated_model is not None
            }
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
            'left_eye': left_eye_tensor.squeeze(0).cpu(),
            'right_eye': right_eye_tensor.squeeze(0).cpu(),
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
        
        dummy_left = torch.randn(3, 224, 224)
        dummy_right = torch.randn(3, 224, 224)
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

@app.route('/calibration/finish', methods=['POST'])
def finish_calibration():
    """Finish calibration by training a personalized model"""
    global calibration_data, calibrated_model, is_calibration_active, model
    if not is_calibration_active:
        return jsonify({
            'error': 'No active calibration session'
        }), 400
    
    if len(calibration_data) < 10:
        return jsonify({
            'error': f'Insufficient calibration data. Need at least 10 samples, have {len(calibration_data)}'
        }), 400
    
    try:
        logger.info(f"Starting calibration training...")

        data = request.json or {}
        epochs = data.get('epochs', 20)
        learning_rate = data.get('learning_rate', 1e-7)
        beta = data.get('beta', 0.5)
        training_start = datetime.now()
        training_result = train_calibrated_model(
            calibration_data,
            model,
            device,
            epochs=epochs,
            lr=learning_rate,
            beta=beta
        )
        training_time = (datetime.now() - training_start).total_seconds()
        calibrated_model = training_result['model']
        is_calibration_active = False

        logger.info(f"Calibration completed successfully!")
        logger.info(f"Training time: {training_time:.1f} seconds")
        logger.info(f"Calibrated model is now active")

        return jsonify({
            'status': 'success',
            'message': 'Calibration completed successfully',
            'training_info': {
                'samples_used': training_result['samples_used'],
                'epochs_trained': training_result['epochs_trained'],
                'final_gaze_loss': training_result['final_loss'],
                'final_mse_loss': training_result['final_mse_loss'],
                'best_gaze_loss': training_result['best_loss'],
                'avg_pixel_loss': round(training_result['avg_pixel_error'], 1),
                'avg_euclidean_error': round(training_result['avg_euclidean_error'], 4),
                'training_time_seconds': round(training_time, 1)
            },
            'model_status': {
                'has_calibrated_model': True,
                'active_model': 'calibrated',
                'can_switch_to_original': True
            }
        })
    
    except Exception as e:
        logger.error(f"Error during calibration training: {e}")
        logger.error(traceback.format_exc())

        is_calibration_active = False
        calibrated_model = None

        return jsonify({
            'error': f'Calibration training failed: {str(e)}'
        }), 500

@app.route('/model/switch', methods=['POST'])
def switch_model():
    global model, calibrated_model
    data = request.json or {}
    target_model = data.get('model_type', 'calibrated')
    if target_model == 'original':
        try:
            original_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(original_checkpoint['model_state_dict'])
            model.eval()
            active_model_type = 'original'
            logger.info("Switched to original model")
        except Exception as e:
            return jsonify({
                'error': f'Failed to load original model: {str(e)}'
            }), 500
    elif target_model == 'calibrated':
        if calibrated_model is None:
            return jsonify({
                'error': 'No calibrated model available'
            }), 400
        model = calibrated_model
        active_model_type = 'calibrated'
        logger.info("Switched to calibrated model")
    else:
        return jsonify({
            'error': 'Invalid model_type. Use "original" or "calibrated"'
        }), 400

    return jsonify({
        'status': 'success',
        'message': f'Switched to {target_model} model',
        'active_model': active_model_type,
        'available_models': {
            'original': True,
            'calibrated': calibrated_model is not None
        }
    })

if __name__ == '__main__':
    logger.info('Starting Flask server...')
    app.run(host='0.0.0.0', port=5000, debug=True)