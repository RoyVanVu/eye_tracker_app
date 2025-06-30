import { useState, useCallback } from 'react';
import axios from 'axios';

const API_URL = "http://127.0.0.1:5000";

export const useCalibration = () => {
    const [calibrationState, setCalibrationState] = useState('inactive');
    const [samplesCollected, setSamplesCollected] = useState(0);
    const [maxSamples, setMaxSamples] = useState(50);
    const [progress, setProgress] = useState(0);
    const [canFinish, setCanFinish] = useState(false);
    const [isCompleted, setIsCompleted] = useState(false);
    const [activeModel, setActiveModel] = useState('original');
    const [hasCalibrated, setHasCalibrated] = useState(false);
    const [trainingResults, setTrainingResults] = useState(null);
    const [lastError, setLastError] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const startCalibration = useCallback(async () => {
        try {
            setIsLoading(true);
            setLastError(null);
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
            setLastError(`Failed to start calibration: ${errorMessage}`);
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
            setLastError(`Failed to add sample: ${errorMessage}`);
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

        try {
            setIsLoading(true);
            setCalibrationState('training');
            setLastError(null);
            console.log("Starting calibration training...");

            const params = {
                epochs: 20,
                learning_rate: 1e-6,
                beta: 0.5,
                ...trainingParams
            };

            const response = await axios.post(`${API_URL}/calibration/finish`, params, {
                timeout: 60000,
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
                console.log("Training results:", response.data.training_info);

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
            setLastError(`Training failed: ${errorMessage}`);
            setCalibrationState('active');
            return {
                success: false,
                error: errorMessage
            };
        } finally {
            setIsLoading(false);
        }
    }, [calibrationState, canFinish]);

    const resetCalibration = useCallback(async () => {
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
                setLastError(null);
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
            setLastError(`Failed to reset: ${errorMessage}`);
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
            console.warn("Cannot switch to calibrated model - no calibration model avaliable");
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
            setLastError(`Failed to switch model: ${errorMessage}`);
            return {
                success: false,
                error: errorMessage
            };
        } finally {
            setIsLoading(false);
        }
    }, [hasCalibrated]);

    const getCalibrationStatus = useCallback(async () => {
        try {
            const response = await axios.get(`${API_URL}/calibration/status`);
            const status = response.data;

            setSamplesCollected(status.samples_collected || 0);
            setMaxSamples(status.max_samples || 50);
            setProgress(status.progress || 0);
            setCanFinish(status.can_finish || false);
            setHasCalibrated(status.has_calibrated_model || false);
            if (status.is_active && calibrationState === 'inactive') {
                setCalibrationState('active');
            } else if (!status.is_active && calibrationState === 'active') {
                setCalibrationState('inactive');
            }

            return {
                success: true,
                status
            };
        } catch (error) {
            console.error("Error getting calibration status:", error);
            return {
                success: false,
                error: error.message
            };
        }
    }, [calibrationState]);

    return {
        calibrationState,
        samplesCollected,
        maxSamples,
        progress,
        canFinish,
        isCompleted,
        activeModel,
        hasCalibrated,
        trainingResults,
        lastError,
        isLoading,

        startCalibration,
        addCalibrationSample,
        finishCalibration,
        resetCalibration,
        switchModel,
        getCalibrationStatus,

        progressPercent: Math.round(progress * 100),
        samplesRemaining: Math.max(0, 10 - samplesCollected),
        isActive: calibrationState === 'active',
        isTraining: calibrationState === 'training',
        isCompleted: calibrationState === 'completed',
    };
};