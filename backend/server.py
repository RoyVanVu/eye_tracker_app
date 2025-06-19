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