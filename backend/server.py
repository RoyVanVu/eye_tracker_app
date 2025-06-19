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
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'model_loaded': True,
        'predictions_made': prediction_counter
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    logger.info("Test endpoint called")
    return jsonify({
        'message': 'Server is working!',
        'device': str(device),
        'predictions_made': prediction_counter
    })

if __name__ == '__main__':
    logger.info('Starting Flask server...')
    app.run(host='0.0.0.0', port=5000, debug=True)