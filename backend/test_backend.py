import requests 
import base64
import json 
from PIL import Image 
import io
import os 

def image_file_to_base64(image_path):
    try:
        with open(image_path, 'rb') as image_file:
            img_data = image_file.read()
            img_str = base64.b64encode(img_data).decode('utf-8')
            img = Image.open(image_path)
            format_lower = img.format.lower()
            return f"data:image/{format_lower};based64,{img_str}"
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def main():
    left_eye_path = "/app/photos/0001Left.jpg"
    right_eye_path = "/app/photos/0001Right.jpg"

    if os.path.exists(left_eye_path) and os.path.exists(right_eye_path):
        left_eye_base64 = image_file_to_base64(left_eye_path)
        right_eye_base64 = image_file_to_base64(right_eye_path)
    else:
        print("Image files not found, using dummy images...")
    
    payload = {
        "leftEye": left_eye_base64,
        "rightEye": right_eye_base64
    }
    try:
        response = requests.post(
            "http://localhost:5000/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()