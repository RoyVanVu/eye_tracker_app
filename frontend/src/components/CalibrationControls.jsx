import React from 'react';

const CalibrationControls = ({
    calibrationState,
    samplesCollected,
    maxSamples,
    progressPercent,
    canFinish,
    isLoading,

    activeModel,
    hasCalibrated,
    trainingResults,

    lastError,

    onStartCalibration,
    onFinishCalibration,
    onResetCalibration,
    onSwitchModel,
}) => {
    const handleStartCalibration = () => {
        if (isLoading) return;
        onStartCalibration();
    };
    
    const handleFinishCalibration = () => {
        if (isLoading || !canFinish) return;
        onFinishCalibration();
    };

    const handleResetCalibration = () => {
        if (isLoading) return;
        onResetCalibration();
    };

    const handleSwitchModel = (modelType) => {
        if (isLoading) return;
        onSwitchModel(modelType);
    };

    const getStatusColor = () => {
        switch (calibrationState) {
            case 'active':
                return '#ff9500'; // Orange
            case 'training':
                return '#007bff'; // Blue
            case 'completed':
                return '#28a745'; // Green
            default:
                return '#6c757d'; // Gray
        }
    };

    const getStatusText = () => {
        switch (calibrationState) {
            case 'active':
                return `Collecting samples... (${samplesCollected}/${maxSamples})`;
            case 'training':
                return 'Training model...';
            case 'completed':
                return 'Calibration completed!';
            default:
                return 'Ready to calibrate';
        }
    };

    return (
        <div style={{
            position: 'fixed',
            bottom: '20px',
            left: '50%',
            transform: 'translateX(-50%)',
            backgroundColor: 'rgba(0, 0, 0, 0.9)',
            color: 'white',
            padding: '20px',
            borderRadius: '12px',
            fontFamily: 'Arial, sans-serif',
            zIndex: 1000,
            minWidth: '400px',
            maxWidth: '600px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)'
        }}>
            {/* Headers */}
            <div style={{
                textAlign: 'center',
                marginBottom: '15px',
                fontSize: '18px',
                fontWeight: 'bold'
            }}>
                Eye Tracking Calibration
            </div>

            {/* Status Section */}
            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '15px',
                padding: '10px',
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                borderRadius: '8px'
            }}>
                <div>
                    <div style={{
                        fontSize: '14px',
                        color: '#ccc'
                    }}>
                        Status:
                    </div>
                    <div style={{
                        color: getStatusColor(),
                        fontWeight: 'bold',
                        fontSize: '16px'
                    }}>
                        {getStatusText()}
                    </div>
                </div>
                <div>
                    <div style={{
                        fontSize: '14px',
                        color: '#ccc'
                    }}>
                        Active Model:
                    </div>
                    <div style={{
                        color: activeModel === 'calibrated' ? '#28a745' : '#ffc107',
                        fontWeight: 'bold',
                        fontSize: '16px'
                    }}>
                        {activeModel === 'calibrated' ? 'Calibrated' : 'Original'}
                    </div>
                </div>
            </div>

            {/* Progress Bar (when calibrating) */}
            {calibrationState === 'active' && (
                <div style={{
                    marginBottom: '15px'
                }}>
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        marginBottom: '5px',
                        fontSize: '14px',
                    }}>
                        <span>Progress</span>
                        <span>{samplesCollected}/{maxSamples}</span>
                    </div>
                    <div style={{
                        width: '100%',
                        height: '8px',
                        backgroundColor: 'rgba(255, 255, 255, 0.2)',
                        borderRadius: '4px',
                        overflow: 'hidden'
                    }}>
                        <div style={{
                            width: `${progressPercent}%`,
                            height: '100%',
                            backgroundColor: '#28a745',
                            transition: 'width 0.3s ease',
                        }} />
                    </div>
                    <div style={{
                        fontSize: '12px',
                        color: '#ccc',
                        marginTop: '5px',
                        textAlign: 'center'
                    }}>
                        {canFinish ? 'Ready to finish!' : `Need ${10 - samplesCollected} more samples to finish`}
                    </div>
                </div>
            )}

            {/* Training Results */}
            {calibrationState === 'completed' && trainingResults && (
                <div style={{
                    marginBottom: '15px',
                    padding: '10px',
                    backgroundColor: 'rgba(40, 167, 69 0.2)',
                    borderRadius: '8px',
                    border: '1px solid rgba(40, 167, 69, 0.4)',
                }}>
                    <div style={{
                        fontSize: '14px',
                        fontWeight: 'bold',
                        marginBottom: '8px'
                    }}>
                        Training Results:
                    </div>
                    <div style={{
                        fontSize: '12px',
                        lineHeight: '1.4'
                    }}>
                        <div>. Samples used: {trainingResults.samplesUsed}</div>
                        <div>. Training time: {trainingResults.training_time_seconds}s</div>
                        <div>. Final loss: {trainingResults.final_gaze_loss?.toFixed(4)}</div>
                        <div>. Pixel error: {trainingResults.avg_pixel_loss?.toFixed(1)} pixels</div>
                    </div>
                </div>
            )}

            {/* Error Display */}
            {lastError && (
                <div style={{
                    marginBottom: '15px',
                    padding: '10px',
                    backgroundColor: 'rgba(220, 53, 69, 0.2)',
                    borderRadius: '8px',
                    border: '1px solid rgba(220, 53, 69, 0.4)',
                    color: '#ff6b6b'
                }}>
                    <div style={{
                        fontSize: '14px',
                        fontWeight: 'bold',
                    }}>Error:</div>
                    <div style={{
                        fontSize: '12px',
                        marginTop: '5px'
                    }}>{lastError}</div>
                </div>
            )}

            {/* Control Buttons */}
            <div style={{
                display: 'flex',
                gap: '10px',
                flexWrap: 'wrap',
                justifyContent: 'center'
            }}>
                {/* Start Calibration */}
                {calibrationState === 'inactive' && (
                    <button 
                        onClick={handleStartCalibration}
                        disabled={isLoading}
                        style={{
                            padding: '12px 20px',
                            backgroundColor: '#007bff',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            fontSize: '14px',
                            fontWeight: 'bold',
                            cursor: isLoading ? 'not-allowed' : 'pointer',
                            opacity: isLoading ? 0.6 : 1,
                            transition: 'all 0.2s ease',
                        }}
                        onMouseOver={(e) => !isLoading && (e.target.style.backgroundColor = '#0056b3')}
                        onMouseOut={(e) => !isLoading && (e.target.style.backgroundColor = '#007bff')}
                    >
                        {isLoading ? 'Starting...' : 'Start Calibration'}
                    </button>
                )}

                {/* Finish Calibration */}
                {calibrationState === 'active' && canFinish && (
                    <button
                        onClick={handleFinishCalibration}
                        disabled={isLoading}
                        style={{
                            padding: '12px 20px',
                            backgroundColor: '#28a745',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            fontSize: '14px',
                            fontWeight: 'bold',
                            cursor: isLoading ? 'not-allowed' : 'pointer',
                            opacity: isLoading ? 0.6 : 1,
                            transition: 'all 0.2s ease',
                        }}
                        onMouseOver={(e) => !isLoading && (e.target.style.backgroundColor = '#1e7e34')}
                        onMouseOut={(e) => !isLoading && (e.target.style.backgroundColor = '#28a745')}
                    >
                        {isLoading ? 'Training...' : 'Finish & Train'}
                    </button>
                )}

                {/* Reset Button */}
                {(calibrationState === 'active' || calibrationState === 'completed') && (
                    <button
                        onClick={handleResetCalibration}
                        disabled={isLoading}
                        style={{
                            padding: '12px 20px',
                            backgroundColor: '#dc3545',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            fontSize: '14px',
                            fontWeight: 'bold',
                            cursor: isLoading ? 'not-allowed' : 'pointer',
                            opacity: isLoading ? 0.6 : 1,
                            transition: 'all 0.2s ease',
                        }}
                        onMouseOver={(e) => !isLoading && (e.target.style.backgroundColor = '#c82333')}
                        onMouseOut={(e) => !isLoading && (e.target.style.backgroundColor = '#dc3545')}
                    >
                        {isLoading ? 'Resetting...' : 'Reset'}
                    </button>
                )}

                {/* Switch Model */}
                {hasCalibrated && calibrationState !== 'active' && calibrationState !== 'training' && (
                    <>
                        <button
                            onClick={() => handleSwitchModel('original')}
                            disabled={isLoading || activeModel === 'original'}
                            style={{
                                padding: '12px 20px',
                                backgroundColor: activeModel === 'original' ? '#6c757d' : '#ffc107',
                                color: 'white',
                                border: 'none',
                                borderRadius: '6px',
                                fontSize: '14px',
                                fontWeight: 'bold',
                                cursor: (isLoading || activeModel === 'original') ? 'not-allowed' : 'pointer',
                                opacity: (isLoading || activeModel === 'original') ? 0.6 : 1,
                                transition: 'all 0.2s ease',
                            }}
                            onMouseOver={(e) => activeModel !== 'original' && !isLoading && (e.target.style.backgroundColor = '#d39e00')}
                            onMouseOut={(e) => activeModel !== 'original' && !isLoading && (e.target.style.backgroundColor = '#ffc107')}
                        >
                            Orignal Model
                        </button>

                        <button
                            onClick={() => handleSwitchModel('calibrated')}
                            disabled={isLoading || activeModel === 'calibrated'}
                            style={{
                                padding: '12px 20px',
                                backgroundColor: activeModel === 'calibrated' ? '#6c757d' : '#17a2b8',
                                color: 'white',
                                border: 'none',
                                borderRadius: '6px',
                                fontSize: '14px',
                                fontWeight: 'bold',
                                cursor: (isLoading || activeModel === 'calibrated') ? 'not-allowed' : 'pointer',
                                opacity: (isLoading || activeModel === 'calibrated') ? 0.6 : 1,
                                transition: 'all 0.2s ease',
                            }}
                            onMouseOver={(e) => activeModel !== 'calibrated' && !isLoading && (e.target.style.backgroundColor = '#138496')}
                            onMouseOut={(e) => activeModel !== 'calibrated' && !isLoading && (e.target.style.backgroundColor = '#17a2b8')}
                        >
                            Calibrated Model
                        </button>
                    </>
                )}
            </div>

            {/* Instructions */}
            <div style={{
                marginTop: '15px',
                fontSize: '12px',
                color: '#999',
                textAlign: 'center',
                lineHeight: '1.4'
            }}>
                {calibrationState === 'inactive' && 'Start calibration to personalize eye tracking accuracy'}
                {calibrationState === 'active' && 'Look at the red dots as they appear . Press ESC to cancel'}
                {calibrationState === 'training' && 'Training your personalized model... Please wait'}
                {calibrationState === 'completed' && 'Calibration complete! Switch between models to compare accuracy'}
            </div>
        </div>
    );
};

export default CalibrationControls;