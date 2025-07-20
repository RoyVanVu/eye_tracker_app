import React, { useRef, useState, useEffect, useDeferredValue } from "react";
import * as tf from "@tensorflow/tfjs";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import { drawMesh } from "../utilities";
import { useEyeTracking } from "../hooks/useEyeTracking";

import WebcamComponent from "./WebcamComponent";
import EyeCanvas from "./EyeCanvas";
import DebugInfo from "./DebugInfo";
import GazeIndicator from "./GazeIndicator";
import CalibrationControls from "./CalibrationControls";
import CalibrationDot from "./CalibrationDot";

function FaceDetection() {
    const webcamRef = useRef(null);
    const canvasRef = useRef(null);
    const leftEyeCanvasRef = useRef(null);
    const rightEyeCanvasRef = useRef(null);
    const [faceDetected, setFaceDetected] = useState(false);
    const [lastError, setLastError] = useState(null);
    const [detector, setDetector] = useState(null);

    const {
        gazeCoordinates,
        isProcessing,
        predictionCount,
        predictGaze,

        calibrationState,
        samplesCollected,
        maxSamples,
        progressPercent,
        canFinish,
        isLoading,
        activeModel,
        hasCalibrated,
        trainingResults,
        startCalibration,
        addCalibrationSample,
        finishCalibration,
        resetCalibration,
        switchModel,
        isActive: isCalibrationActive,

        serverStatus,
        isConnected,
        
        lastError: eyeTrackingError,
        gazeError,
        calibrationError,
    } = useEyeTracking();

    console.log("Calibration Debug:", {
        calibrationState,
        isCalibrationActive,
        samplesCollected,
        maxSamples,
        progressPercent,
        canFinish,
        timestamp: new Date().toISOString()
    });

    console.log("Component: Current hook values:", {
        serverStatus,
        isConnected,
        timestamp: new Date().toISOString()
    });

    const combinedError = lastError || eyeTrackingError;

    const initializeFaceMesh = async () => {
        try {
            console.log("Loading Face Landmarks model...");

            const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
            const detectorConfig = {
                runtime: 'tfjs',
                refineLandmarks: true,
            };

            const newDetector = await faceLandmarksDetection.createDetector(model, detectorConfig);
            console.log("Face Landmarks model loaded successfully");

            setDetector(newDetector);
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
                    // console.log(`Face detected with ${face[0].keypoints.length} keypoints`);

                    const ctx = canvasRef.current.getContext("2d");
                    drawMesh(face, ctx);
                    const keypoints = face[0].keypoints;
                    const leftEyeIndices = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155];
                    const rightEyeIndices = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382];
                    const leftBox = getEyeBox(keypoints, leftEyeIndices);
                    const rightBox = getEyeBox(keypoints, rightEyeIndices);
                    // console.log("Eye boxes:", { leftBox, rightBox });

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

                    if (leftBox.width > 0 && rightBox.width > 0) {
                        console.log("Detection check:", {
                            isProcessing,
                            isConnected,
                            serverStatus,
                            isCalibrationActive,
                            leftBoxWidth: leftBox.width,
                            rightBoxWidth: rightBox.width,
                            timestamp: new Date().toISOString()
                        });

                        if (!isProcessing && isConnected) {
                            console.log("Running gaze prediction...");
                            predictGaze(leftEyeCanvasRef.current, rightEyeCanvasRef.current);
                        } else {
                            console.log("Skipping prediction:", {
                                isProcessing,
                                isConnected,
                                serverStatus,
                                reason: isProcessing ? "Still processing" : "Server not connected",
                                timestamp: new Date().toISOString()
                            });
                        }

                        if (isCalibrationActive) {
                            console.log("Calibration mode - eye images ready for capture");
                        }
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

    const handleCalibrationCapture = async (targetX, targetY) => {
        console.log("handleCalibrationCapture called!");
        console.log(`Capturing calibration sample at (${targetX.toFixed(3)}, ${targetY.toFixed(3)})`);
        console.log("Current state:", {
            isCalibrationActive,
            calibrationState,
            faceDetected,
            samplesCollected,
            leftEyeCanvas: !!leftEyeCanvasRef.current,
            rightEyeCanvas: !!rightEyeCanvasRef.current
        });
        
        if (!leftEyeCanvasRef.current || !rightEyeCanvasRef.current) {
            console.error("Eye canvases not ready for calibration");
            return;
        }
        if (!faceDetected) {
            console.warn("No face detected - skipping calibration sample");
            return;
        }

        try {
            console.log("Calling addCalibrationSample...")
            const result = await addCalibrationSample(
                leftEyeCanvasRef.current,
                rightEyeCanvasRef.current,
                targetX,
                targetY
            );
            console.log("addCalibrationSample result:", result);

            if (result.success) {
                console.log(`Calibration sample added successfully! Progress: ${(result.progress * 100).toFixed(1)}%`);
            } else {
                console.error("Failed to add calibration sample:", result.error);
            }
        } catch (error) {
            console.error("Error during calibration capture:", error);
        }
    };

    const handleCalibrationComplete = async () => {
        console.log("Calibration sample collection completed!");

        try {
            const result = await finishCalibration();
            if (result.success) {
                console.log("Calibration training completed successfully!");
                console.log("Training results:", result.trainingInfo);
            } else {
                console.error("Calibration training failed:", result.error);
            }
        } catch (error) {
            console.error("Error finishing calibration:", error);
        }
    };

    const handleCalibrationCancel = async () => {
        console.log("Calibration cancelled by user");

        try {
            await resetCalibration();
            console.log("Calibration reset successfully");
        } catch (error) {
            console.error("Error resetting calibration:", error);
        }
    };

    useEffect(() => {
        console.log("Initializing face detector...");
        initializeFaceMesh();
    }, []);

    useEffect(() => {
        if (!detector) {
            console.log("Detector not ready yet...");
            return;
        }
        console.log("Setting up detection loop with current state:", {
            isConnected,
            serverStatus,
            isProcessing,
            isCalibrationActive,
            timestamp: new Date().toISOString()
        });

        const intervalID = setInterval(() => {
            detect(detector);
        }, 200);

        return () => {
            console.log("Cleaning up detection interval");
            clearInterval(intervalID);
        };
    }, [detector, isConnected, serverStatus, isProcessing, isCalibrationActive, predictGaze]);

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

            <CalibrationDot 
                isActive={isCalibrationActive}
                onCaptureSample={handleCalibrationCapture}
                onComplete={handleCalibrationComplete}
                onCancel={handleCalibrationCancel}
                samplesCollected={samplesCollected}
                maxSamples={maxSamples}
                progressPercent={progressPercent}
            />

            <DebugInfo 
                serverStatus={serverStatus}
                faceDetected={faceDetected}
                isProcessing={isProcessing}
                predictionCount={predictionCount}
                gazeCoordinates={gazeCoordinates}
                lastError={combinedError}
                calibrationState={calibrationState}
                activeModel={activeModel}
                samplesCollected={samplesCollected}
            />

            <GazeIndicator 
                gazeCoordinates={gazeCoordinates}
                serverStatus={serverStatus}
                isCalibrationActive={isCalibrationActive}
                activeModel={activeModel}
            />

            <CalibrationControls 
                calibrationState={calibrationState}
                samplesCollected={samplesCollected}
                maxSamples={maxSamples}
                progressPercent={progressPercent}
                canFinish={canFinish}
                isLoading={isLoading}
                activeModel={activeModel}
                hasCalibrated={hasCalibrated}
                trainingResults={trainingResults}
                lastError={calibrationError}
                onStartCalibration={async () => {
                    console.log("Start Calibration button clicked!");
                    const result = await startCalibration();
                    console.log("Start Calibration result:", result);
                    return result;
                }}
                onFinishCalibration={handleCalibrationComplete}
                onResetCalibration={handleCalibrationCancel}
                onSwitchModel={switchModel}
            />

            {/* DEBUG: Test calibration button */}
            {isCalibrationActive && (
                <button
                    onClick={() => {
                        console.log("🧪 TEST BUTTON: Manual calibration trigger!");
                        handleCalibrationCapture(0.5, 0.5); 
                    }}
                    style={{
                        position: 'fixed',
                        top: '20px',
                        right: '20px',
                        padding: '10px 20px',
                        backgroundColor: 'red',
                        color: 'white',
                        border: 'none',
                        borderRadius: '5px',
                        fontSize: '14px',
                        fontWeight: 'bold',
                        cursor: 'pointer',
                        zIndex: 9999
                    }}
                >
                    🧪 TEST CAPTURE
                </button>
            )}
        </div>
    );
}

export default FaceDetection;