import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import { drawMesh } from "../utilities";
import axios from "axios"
import WebcamComponent from "./WebcamComponent";
import EyeCanvas from "./EyeCanvas";

const API_URL = "http://127.0.0.1:5000";

function FaceDetection() {
    const webcamRef = useRef(null);
    const canvasRef = useRef(null);
    const leftEyeCanvasRef = useRef(null);
    const rightEyeCanvasRef = useRef(null);
    const [faceDetected, setFaceDetected] = useState(false);
    const [lastError, setLastError] = useState(null);
    const [gazeCoordinates, setGazeCoordinates] = useState({ x: 0, y: 0 });
    const [isProcessing, setIsProcessing] = useState(false);
    const [serverStatus, setServerStatus] = useState("Connecting...");
    const [predictionCount, setPredictionCount] = useState(0);

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

    const canvasToBase64 = (canvas) => {
        try {
            if(!canvas || canvas.width === 0 || canvas.height === 0) {
                console.error("Canvas is invalid or has zero dimensions");
                return null;
            }
            return canvas.toDataURL('image/jpeg', 0.8);
        } catch (error) {
            console.error("Error converting canvas to base64:", error);
            return null;
        }
    };

    const predictGaze = async (leftEyeCanvas, rightEyeCanvas) => {
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
                setLastError(null)
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
    };

    const runFaceMesh = async () => {
        try {
            console.log("Loading Face Landmarks model...");
            
            const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
            const detectorConfig = {
                runtime: 'tfjs',
                refineLandmarks: true,
            };
            
            const detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
            console.log("Face Landmarks model loaded successfully");
            
            setInterval(() => {
                detect(detector);
            }, 200);
        } catch (error) {
            console.log("Error loading Face Landmarks:", error);
            setLastError(`Face Landmarks loading error: ${error.message}`);
        }
    };

    const detect = async (net) => {
        if (
            typeof webcamRef.current !== "undefined" &&
            webcamRef.current !== null &&
            webcamRef.current.video.readyState === 4
        ) {
            try {
                const video = webcamRef.current.video;
                const videoWidth = webcamRef.current.video.videoWidth;
                const videoHeight = webcamRef.current.video.videoHeight;

                webcamRef.current.video.width = videoWidth;
                webcamRef.current.video.height = videoHeight;
                canvasRef.current.width = videoWidth;
                canvasRef.current.height = videoHeight;

                const face = await net.estimateFaces(video);
                setFaceDetected(face.length > 0);

                if (face.length > 0) {
                    console.log(`Face detected with ${face[0].keypoints.length} keypoints`);

                    const ctx = canvasRef.current.getContext("2d");
                    drawMesh(face, ctx);
                    const keypoints = face[0].keypoints;
                    const leftEyeIndices = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155];
                    const rightEyeIndices = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382];
                    const leftBox = getEyeBox(keypoints, leftEyeIndices);
                    const rightBox = getEyeBox(keypoints, rightEyeIndices);
                    console.log("Eye boxes:", { leftBox, rightBox });

                    if (leftEyeCanvasRef.current.width === 0) {
                        leftEyeCanvasRef.current.width = 224;
                        leftEyeCanvasRef.current.height = 224;
                    }
                    if (rightEyeCanvasRef.current.width === 0) {
                        rightEyeCanvasRef.current.width = 224;
                        rightEyeCanvasRef.current.height = 224;
                    }

                    const ctxLeft = leftEyeCanvasRef.current.getContext("2d");
                    ctxLeft.clearRect(0, 0, leftEyeCanvasRef.current.width, leftEyeCanvasRef.current.height);
                    ctxLeft.drawImage(
                        video, 
                        leftBox.x,
                        leftBox.y,
                        leftBox.width,
                        leftBox.height,
                        0, 0,
                        leftEyeCanvasRef.current.width,
                        leftEyeCanvasRef.current.height
                    );

                    const ctxRight = rightEyeCanvasRef.current.getContext("2d");
                    ctxRight.clearRect(0, 0, rightEyeCanvasRef.current.width, rightEyeCanvasRef.current.height);
                    ctxRight.drawImage(
                        video,
                        rightBox.x,
                        rightBox.y,
                        rightBox.width,
                        rightBox.height,
                        0, 0, 
                        rightEyeCanvasRef.current.width,
                        rightEyeCanvasRef.current.height
                    );

                    console.log("Prediction conditions check:", {
                        isProcessing, 
                        serverStatus, 
                        leftBoxValid: leftBox.width > 0,
                        rightBoxValid: rightBox.width > 0,
                        leftBoxWidth: leftBox.width,
                        rightBoxWidth: rightBox.width
                    });
                    
                    if (!isProcessing && leftBox.width > 0 && rightBox.width > 0) {
                        console.log("All condition met - Calling predictGaze...");
                        predictGaze(leftEyeCanvasRef.current, rightEyeCanvasRef.current);
                    } else {
                        console.log("Condition not met - skipping prediction");
                    }
                } else {
                    console.log("No face detected");
                }
            } catch (error) {
                console.log("Error in face detection:", error);
                setLastError(`Detection error: ${error.message}`);
            }
        }
    };

    const getEyeBox = (keypoints, indices) => {
        const points = indices.map(i => keypoints[i]);
        const xs = points.map(p => p.x);
        const ys = points.map(p => p.y);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        const padding = 10;
        const width = maxX - minX + padding * 2;
        const height = maxY - minY + padding * 2;

        return {
            x: Math.max(0, minX - padding),
            y: Math.max(0, minY - padding),
            width: width,
            height: height
        };
    };

    useEffect(() => {
        runFaceMesh();
        return () => {
            // if (sessionTimerRef.current) clearInterval(sessionTimerRef.current);
            // if (sessionTimerRef.current) clearInterval(sessionTimerRef.current);
        };
    }, []);

    return (
        <div className="face-detection-container">
            <WebcamComponent ref={webcamRef} />

            <canvas 
                ref={canvasRef}
                style={{
                    position: "absolute",
                    marginLeft: "auto",
                    marginRight: "auto",
                    left: 0,
                    right: 0,
                    textAlign: "center",
                    width: 640,
                    height: 640,
                    opacity: 0
                }}
            />

            <EyeCanvas 
                ref={leftEyeCanvasRef}
                position="left"
                marginLeft="10px"
            />

            <EyeCanvas 
                ref={rightEyeCanvasRef}
                position="right"
                marginLeft="170px"
            />

            <div style={{
                position: "absolute",
                bottom: "20px",
                left: "20px",
                background: "rgba(0, 0, 0, 0.8)",
                color: "white",
                padding: "15px",
                borderRadius: "8px",
                zIndex: 9,
                fontFamily: "monospace",
                fontSize: "12px",
                minWidth: "300px",
            }}>
                <div><strong>Debug Info:</strong></div>
                <div>Server Status: 
                    <span style={{
                        color: serverStatus === "Connected" ? "lime" : 
                                serverStatus === "Connecting..." ? "yellow" : "red",
                        marginLeft: "5px",
                        fontWeight: "bold"
                    }}>
                        {serverStatus}
                    </span>
                </div>
                <div>Face Detected:
                    <span style={{
                        color: faceDetected ? "lime" : "orange", 
                        marginLeft: "5px"
                    }}>
                        {faceDetected ? "YES" : "NO"}
                    </span>
                </div>
                <div>Processing:
                    <span style={{
                        color: isProcessing ? "yellow" : "lime", 
                        marginLeft: "5px"
                    }}>
                        {isProcessing ? "YES" : "NO"}
                    </span>
                </div>
                <div>Prediction Made: <span style={{ color: "cyan" }}>{predictionCount}</span></div>
                <div>Gaze X: <span style={{ color: "lime" }}>{gazeCoordinates.x?.toFixed(4) || "N/A"}</span></div>
                <div>Gaze Y: <span style={{ color: "lime" }}>{gazeCoordinates.y?.toFixed(4) || "N/A"}</span></div>
                {lastError && (
                    <div style={{
                        color: "red",
                        marginTop: "5px",
                        fontSize: "10px"
                    }}>
                        Error: {lastError}
                    </div>
                )}
            </div>

            {serverStatus === "Connected" && gazeCoordinates.x !== undefined && gazeCoordinates.x !== 0 && (
                <div style={{
                    position: "absolute",
                    left: `${gazeCoordinates.x * window.innerWidth}px`,
                    top: `${gazeCoordinates.y * window.innerHeight}px`,
                    width: "20px",
                    height: "20px",
                    borderRadius: "50%",
                    backgroundColor: "red",
                    transform: "translate(-50%, -50%)",
                    zIndex: 15,
                    boxShadow: "0 0 10px rgba(255, 0, 0, 0.8)",
                }} />
            )}
        </div>
    );
}

export default FaceDetection;