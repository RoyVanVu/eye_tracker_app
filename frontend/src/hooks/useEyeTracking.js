import { useState, useEffect, useCallback } from "react";
import axios from 'axios';

const API_URL = "http://127.0.0.1:5000";

export const useEyeTracking = () => {
    const [gazeCoordinates, setGazeCoordinates] = useState({ x: 0, y: 0});
    const [isProcessing, setIsProcessing] = useState(false);
    const [predictionCount, setPredictionCount] = useState(0);
    const [gazeError, setGazeError] = useState(null);

    const [calibrationState, setCalibrationState] = useState('inactive');
    const [samplesCollected, setSamplesCollected] = useState(0);
    const [maxSamples, setMaxSamples] = useState(50);
    const [progress, setProgress] = useState(0);
    const [canFinish, setCanFinish] = useState(false);
    const [isCompleted, setIsCompleted] = useState(false);
    const [activeModel, setActiveModel] = useState('original');
    const [hasCalibrated, setHasCalibrated] = useState(false);
    const [trainingResults, setTrainingResults] = useState(null);
    const [calibrationError, setCalibrationError] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isFinishing, setIsFinishing] = useState(false);

    const [serverStatus, setServerStatus] = useState("Connecting...");
    const [serverError, setServerError] = useState(null);

    useEffect(() => { 
        const checkServerStatus = async () => {
            try {
                console.log("Frontend: Starting health check...");
                console.log("Frontend: API_URL =", API_URL);
                console.log("Frontend: Full URL =", `${API_URL}/health`);
                
                const response = await axios.get(`${API_URL}/health`, {
                    timeout: 3000
                });
                
                console.log("Frontend: Response received!");
                console.log("Frontend: response.status =", response.status);
                console.log("Frontend: response.data =", response.data);
                console.log("Frontend: typeof response.status =", typeof response.status);
                
                if (response.status === 200) {
                    console.log("Frontend: Status is 200, setting to Connected");
                    setServerStatus("Connected");
                    setServerError(null);
                    console.log("Frontend: Server status set to Connected");
                } else {
                    console.log("Frontend: Status is not 200, got:", response.status);
                    setServerStatus("Disconnected");
                    setServerError(`Unexpected status code: ${response.status}`);
                }
            } catch (error) {
                console.error("Frontend: Health check failed");
                console.error("Frontend: Error type:", error.constructor.name);
                console.error("Frontend: Error message:", error.message);
                console.error("Frontend: Error code:", error.code);
                console.error("Frontend: Error response:", error.response);
                console.error("Frontend: Full error:", error);
                
                setServerStatus("Disconnected");
                setServerError(`Connection error: ${error.message}`);
            }
        };

        console.log("Frontend: Initializing health check...");
        checkServerStatus();
        
        const interval = setInterval(() => {
            console.log("Frontend: Running periodic health check...");
            checkServerStatus();
        }, 5000);

        return () => {
            console.log("Frontend: Cleaning up health check interval");
            clearInterval(interval);
        };
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
            if (!leftEyeBase64 || !rightEyeBase64) {
                console.error("Failed to convert eye image to base64");
                setGazeError("Failed to convert eye images");
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
                setGazeError(null);
                console.log("Prediction successful:", data.gaze);
            } else {
                console.error("Invalid gaze data received:", data);
                setGazeError("Invalid prediction data received");
            }
        } catch (error) {
            console.error("Error predicting gaze:", error);
            setGazeError(`Prediction error: ${error.response?.data?.error || error.message}`);
            
            if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
                setServerStatus("Disconnected");
            }
        } finally {
            setTimeout(() => {
                setIsProcessing(false);
            }, 300);
        }
    }, [isProcessing, canvasToBase64]);

    const startCalibration = useCallback(async () => {
        try {
            setIsLoading(true);
            setCalibrationError(null);
            console.log("Starting calibration session...");

            const response = await axios.post(`${API_URL}/calibration/start`);
            if (response.data.status === 'success') {
                setCalibrationState('active');
                setSamplesCollected(response.data.current_samples || 0);
                setMaxSamples(response.data.max_samples || 50);
                setProgress(0);
                setCanFinish(false);
                setIsCompleted(false);
                setTrainingResults(null);
                console.log("Calibration started successfully");
                return {
                    success: true
                };
            } else {
                throw new Error(response.data.message || 'Failed to start calibration');
            }
        } catch (error) {
            console.error("Error starting calibration:", error);
            const errorMessage = error.response?.data?.error || error.message;
            setCalibrationError(`Failed to start calibration: ${errorMessage}`);
            setCalibrationState('inactive');
            return {
                success: false,
                error: errorMessage
            };
        } finally {
            setIsLoading(false);
        }
    }, []);

    const addCalibrationSample = useCallback(async (leftEyeCanvas, rightEyeCanvas, targetX, targetY) => {
        if (calibrationState !== 'active') {
            console.warn("Cannot add sample - calibration is not active");
            return {
                success: false,
                error: "Calibration is not active"
            };
        }

        try {
            console.log(`Adding calibration sample at target (${targetX.toFixed(3)}, ${targetY.toFixed(3)})`);

            const leftEyeBase64 = leftEyeCanvas.toDataURL('image/jpeg', 0.8);
            const rightEyeBase64 = rightEyeCanvas.toDataURL('image/jpeg', 0.8);
            if (!leftEyeBase64 || !rightEyeBase64) {
                throw new Error("Failed to convert eye images to base64");
            }

            const response = await axios.post(`${API_URL}/calibration/add_sample`, {
                leftEye: leftEyeBase64,
                rightEye: rightEyeBase64,
                targetX: targetX,
                targetY: targetY
            }, {
                timeout: 10000,
                headers: { 
                    'Content-Type': 'application/json'
                }
            });
            if (response.data.status === 'success') {
                setSamplesCollected(response.data.samples_collected);
                setProgress(response.data.progress);
                setCanFinish(response.data.can_finish);
                setIsCompleted(response.data.is_completed);
                console.log(`Sample added successfully. Progress: ${(response.data.progress * 100).toFixed(1)}%`);

                return {
                    success: true,
                    samplesCollected: response.data.samples_collected,
                    progress: response.data.progress,
                    canFinish: response.data.can_finish,
                    isCompleted: response.data.is_completed
                };
            } else {
                throw new Error(response.data.message || 'Failed to add sample');
            }
        } catch (error) {
            console.error("Error adding calibration sample:", error);
            const errorMessage = error.response?.data?.error || error.message;
            setCalibrationError(`Failed to add sample: ${errorMessage}`);
            return {
                success: false,
                error: errorMessage
            };
        }
    }, [calibrationState]);

    const finishCalibration = useCallback(async (trainingParams = {}) => {
        if (calibrationState !== 'active') {
            console.warn("Cannot finish calibration - not in active state");
            return {
                success: false,
                error: "Calibration is not active"
            };
        }

        if (!canFinish) {
            console.warn("Cannot finish calibration - insufficient samples");
            return {
                success: false,
                error: "Need at least 10 samples to finish calibration"
            };
        }

        if (isFinishing) {
            console.log("Calibration finish already in progress, ignoring duplicate call");
            return {
                success: false,
                error: "Already finishing calibration"
            };
        }

        try {
            setIsFinishing(true);
            setIsLoading(true);
            setCalibrationState('training');
            setCalibrationError(null);
            console.log("Starting calibration training...");

            const params = {
                epochs: 50,
                learning_rate: 1e-6,
                beta: 0.5,
                ...trainingParams
            };

            const response = await axios.post(`${API_URL}/calibration/finish`, params, {
                timeout: 300000,
                headers: { 
                    'Content-Type': 'application/json' 
                }
            });
            if (response.data.status === 'success') {
                setCalibrationState('completed');
                setTrainingResults(response.data.training_info);
                setActiveModel('calibrated');
                setHasCalibrated(true);
                console.log("Calibration training completed successfully!");

                return {
                    success: true,
                    trainingInfo: response.data.training_info,
                    modelStatus: response.data.model_status
                };
            } else {
                throw new Error(response.data.message || 'Training failed');
            }
        } catch (error) {
            console.error("Error finishing calibration:", error);
            const errorMessage = error.response?.data?.error || error.message;
            setCalibrationError(`Training failed: ${errorMessage}`);
            setCalibrationState('active');
            return {
                success: false,
                error: errorMessage
            };
        } finally {
            setIsLoading(false);
            setIsFinishing(false);
        }
    }, [calibrationState, canFinish, isFinishing]);

    const resetCalibration =  useCallback(async () => {
        try {
            setIsLoading(true);
            console.log("Resetting calibration...");

            const response = await axios.post(`${API_URL}/calibration/reset`);
            if (response.data.status === 'success') {
                setCalibrationState('inactive');
                setSamplesCollected(0);
                setProgress(0);
                setCanFinish(false);
                setIsCompleted(false);
                setTrainingResults(null);
                setCalibrationError(null);
                console.log("Calibration reset successfully");
                return {
                    success: true
                };
            } else {
                throw new Error(response.data.message || 'Failed to reset calibration');
            }
        } catch (error) {
            console.error("Error resetting calibration:", error);
            const errorMessage = error.response?.data?.error || error.message;
            setCalibrationError(`Failed to reset: ${errorMessage}`);
            return {
                success: false,
                error: errorMessage
            };
        } finally {
            setIsLoading(false);
        }
    }, []);

    const switchModel = useCallback(async (modelType) => {
        if (!['original', 'calibrated'].includes(modelType)) {
            console.warn("Invalid model type:", modelType);
            return {
                success: false,
                error: "Invalid model type"
            };
        }

        if (modelType === 'calibrated' && !hasCalibrated) {
            console.warn("Cannot switch to calibrated model - no calibration model available");
            return {
                success: false,
                error: "No calibrated model available"
            };
        }

        try {
            setIsLoading(true);
            console.log(`Switching to ${modelType} model...`);

            const response = await axios.post(`${API_URL}/model/switch`, {
                model_type: modelType
            });
            if (response.data.status === 'success') {
                setActiveModel(response.data.active_model);
                console.log(`Successfully switched to ${response.data.active_model} model`);
                return {
                    success: true,
                    activeModel: response.data.active_model
                };
            } else {
                throw new Error(response.data.message || 'Failed to switch model');
            }
        } catch (error) {
            console.error("Error switching model:", error);
            const errorMessage = error.response?.data?.error || error.message;
            setCalibrationError(`Failed to switch model: ${errorMessage}`);
            return {
                success: false,
                error: errorMessage
            };
        } finally {
            setIsLoading(false);
        }
    }, [hasCalibrated]);

    return {
        gazeCoordinates,
        isProcessing,
        predictionCount,
        predictGaze,

        calibrationState,
        samplesCollected,
        maxSamples,
        progressPercent: Math.round(progress * 100),
        canFinish,
        isCompleted,
        activeModel,
        hasCalibrated,
        trainingResults,
        isLoading,
        startCalibration,
        addCalibrationSample,
        finishCalibration,
        resetCalibration,
        switchModel,

        serverStatus,
        isConnected: serverStatus === 'Connected',
        isDisconnected: serverStatus === 'Disconnected',
        isConnecting: serverStatus === 'Connecting...',

        lastError: gazeError || calibrationError || serverError,
        gazeError,
        calibrationError,
        serverError,

        isActive: calibrationState === 'active',
        isTraining: calibrationState === 'training',
        samplesRemaining: Math.max(0, 10 - samplesCollected),
    };
};