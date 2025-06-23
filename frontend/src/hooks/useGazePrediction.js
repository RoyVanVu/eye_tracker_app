import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_URL = "http://127.0.0.1:5000";

export const useGazePrediction = () => {
    const [gazeCoordinates, setGazeCoordinates] = useState({ x: 0, y: 0 });
    const [isProcessing, setIsProcessing] = useState(false);
    const [serverStatus, setServerStatus] = useState("Connecting...");
    const [predictionCount, setPredictionCount] = useState(0);
    const [lastError, setLastError] = useState(null);

    useEffect(() => {
        const checkServerStatus = async () => {
            try {
                console.log("Checking server status...");
                const response = await axios.get(`${API_URL}/health`, {
                    timeout: 3000
                });
                console.log("Server response:", response);
                if (response.status === 200) {
                    setServerStatus("Connected");
                    console.log("Backend server is connected:", response.data);
                } else {
                    console.log("Unexpected status code:", response.status);
                    setServerStatus("Disconected");
                }
            } catch (error) {
                console.error("Cannot connect to backend server:", error);
                console.error("Error details:", {
                    message: error.message,
                    code: error.code,
                    response: error.response?.data
                });
                serverStatus("Disconnected");
                setLastError(`Connection error: ${error.message}`);
            }
        };

        checkServerStatus();
        const interval = setInterval(checkServerStatus, 5000);
        return () => clearInterval(interval);
    }, []);

    const canvasToBase64 = useCallback((canvas) => {
        try {
            if (!canvas || canvas.width === 0 || canvas.height === 0) {
                console.error("Canvas is invalid or has zero dimensions");
                return null;
            }
            return canvas.toDataURL('image/jpeg', 0.8);
        } catch (error) {
            console.error("Error converting canvas to base64:", error);
            return null;
        }
    }, []);

    const predictGaze = useCallback(async (leftEyeCanvas, rightEyeCanvas) => {
        console.log("predictGaze function called!");
        if (isProcessing) {
            console.log(`Skipping prediction - Processing: ${isProcessing}`);
            return;
        }

        try {
            setIsProcessing(true);
            console.log("Starting gaze prediction...");
            const leftEyeBase64 = canvasToBase64(leftEyeCanvas);
            const rightEyeBase64 = canvasToBase64(rightEyeCanvas);
            
            console.log("Canvas conversion results:", {
                leftEyeBase64: leftEyeBase64 ? "Success" : "Failed",
                rightEyeBase64: rightEyeBase64 ? "Success" : "Failed",
                leftLength: leftEyeBase64?.length || 0,
                rightLength: rightEyeBase64?.length || 0
            });

            if (!leftEyeBase64 || !rightEyeBase64) {
                console.error("Failed to convert eye image to base64");
                setLastError("Failed to convert eye images");
                return;
            }
            console.log("Sending prediction request to backend...");

            const response = await axios.post(`${API_URL}/predict`, {
                leftEye: leftEyeBase64,
                rightEye: rightEyeBase64
            }, {
                timeout: 5000,
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = response.data;
            console.log("Gaze prediction response:", data);
            if (data.gaze && typeof data.gaze.x === 'number' && typeof data.gaze.y === 'number') {
                setGazeCoordinates(data.gaze);
                setPredictionCount(prev => prev + 1);
                setLastError(null);
            } else {
                console.error("Invalid gaze data received:", data);
                setLastError("Invalid prediction data received");
            }
        } catch (error) {
            console.error("Error prediction gaze:", error);
            setLastError(`Prediction error: ${error.response?.data?.error || error.message}`);
            if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
                setServerStatus("Disconnected");
            }
        } finally {
            setTimeout(() => {
                setIsProcessing(false);
            }, 300);
        }
    }, [isProcessing, canvasToBase64]);

    return {
        gazeCoordinates,
        isProcessing,
        serverStatus,
        predictionCount,
        lastError,
        predictGaze
    };
};