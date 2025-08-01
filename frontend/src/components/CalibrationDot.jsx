import React, { useState, useEffect, useCallback, useRef } from 'react';

const CalibrationDot = ({
    isActive,
    onCaptureSample,
    onComplete,
    onCancel,
    samplesCollected = 0,
    maxSamples = 200,
    progressPercent = 0
}) => {
    const GRID_COLS = 10;
    const GRID_ROWS = 10;
    const TOTAL_GRID_POINTS = GRID_COLS * GRID_ROWS;

    const COUNTDOWN_TIME = 0.5;
    const CAPTURE_DELAY = 300;
    const DOT_TRANSITION_DELAY = 200;
    const COUNTDOWN_INTERVAL = 100;

    const [currentDot, setCurrentDot] = useState({ 
        x: 0.5,
        y: 0.5
    });
    const [currentGridIndex, setCurrentGridIndex] = useState(0);
    const [dotState, setDotState] = useState('waiting');
    const [countdown, setCountdown] = useState(COUNTDOWN_TIME);
    const [isCapturing, setIsCapturing] = useState(false);
    const [lastCapturedPosition, setLastCapturedPosition] = useState(null);
    const [isProcessingSample, setIsProcessingSample] = useState(false);

    const currentDotRef = useRef(currentDot);
    const samplesCollectedRef = useRef(samplesCollected);
    const maxSamplesRef = useRef(maxSamples);
    const onCaptureSampleRef = useRef(onCaptureSample);
    const onCompleteRef = useRef(onComplete);

    useEffect(() => {
        currentDotRef.current = currentDot;
    }, [currentDot]);
    useEffect(() => {
        samplesCollectedRef.current = samplesCollected;
    }, [samplesCollected]);
    useEffect(() => {
        maxSamplesRef.current = maxSamples;
    }, [maxSamples]);
    useEffect(() => {
        onCaptureSampleRef.current = onCaptureSample;
    }, [onCaptureSample]);
    useEffect(() => {
        onCompleteRef.current = onComplete;
    }, [onComplete]);

    useEffect(() => {
        if (samplesCollected >= maxSamples && isActive) {
            console.log(`Maximum samples (${maxSamples}) reached, completing calibration`);
            onComplete();
        }
    }, [samplesCollected, maxSamples, isActive, onComplete]);

    console.log("CalibrationDot render:", {
        isActive,
        samplesCollected,
        dotState,
        countdown,
        currentGridIndex,
        isCapturing,
        isProcessingSample,
    });

    const getGridPosition = useCallback((gridIndex) => {
        const col = Math.floor(gridIndex / GRID_ROWS);
        const row = gridIndex % GRID_ROWS;

        const margin = 0.05;
        const usableWidth = 1 - 2 * margin;
        const usableHeight = 1 - 2 * margin;

        let x, y;
        if (GRID_COLS === 1) {
            x = 0.5;
        } else {
            x = margin + (col / (GRID_COLS - 1)) * usableWidth;
        }

        if (GRID_ROWS === 1) {
            y = 0.5;
        } else {
            y = margin + (row / (GRID_ROWS - 1)) * usableHeight;
        }

        x = Math.max(margin, Math.min(1 - margin, x));
        y = Math.max(margin, Math.min(1 - margin, y));

        return {
            x,
            y
        };
    }, [GRID_COLS, GRID_ROWS]);

    const moveToNextPoint = useCallback(() => {
        if (samplesCollectedRef.current >= maxSamplesRef.current) {
            console.log("Maximum samples reached, stopping calibration");
            onCompleteRef.current();
            return;
        }

        console.log("Moving to next point");

        const nextGridIndex = (currentGridIndex + 1) % TOTAL_GRID_POINTS;
        setCurrentGridIndex(nextGridIndex);

        const newPosition = getGridPosition(nextGridIndex);
        setCurrentDot(newPosition);
        setDotState('waiting');
        setCountdown(COUNTDOWN_TIME);
        setIsCapturing(false);
        setIsProcessingSample(false);
        setLastCapturedPosition(null);

        setTimeout(() => {
            if (samplesCollectedRef.current >= maxSamplesRef.current) {
                console.log("Maximum samples reached during transition, stopping calibration");
                onCompleteRef.current();
                return;
            }

            console.log("Auto-starting next dot sequence");
            setDotState('active');
            setCountdown(COUNTDOWN_TIME);
        }, DOT_TRANSITION_DELAY);
    }, [currentGridIndex, getGridPosition, TOTAL_GRID_POINTS, COUNTDOWN_TIME, DOT_TRANSITION_DELAY]);

    const captureCurrentSample = useCallback(() => {
        if (samplesCollectedRef.current >= maxSamplesRef.current) {
            console.log("Maximum samples already reached, not capturing");
            onCompleteRef.current();
            return;
        }

        if (isCapturing) {
            console.log("Already capturing, ignoring duplicate call");
            return;
        }

        if (isProcessingSample) {
            console.log("Already processing sample, ignoring duplicate call");
            return;
        }

        const currentPos = currentDotRef.current;
        if (lastCapturedPosition &&
            Math.abs(lastCapturedPosition.x - currentPos.x) < 0.001 &&
            Math.abs(lastCapturedPosition.y - currentPos.y) < 0.001
        ) {
            console.log("Same position already captured, ignoring duplicate");
            return;
        }

        console.log("Capturing sample!");
        setIsCapturing(true);
        setIsProcessingSample(true);
        setDotState('capturing');
        setLastCapturedPosition({
            x: currentPos.x,
            y: currentPos.y
        });

        onCaptureSampleRef.current(currentPos.x, currentPos.y);

        setTimeout(() => {
            const currentSamplesCount = samplesCollectedRef.current + 1;

            if (samplesCollectedRef.current + 1 >= maxSamplesRef.current) {
                console.log("Calibration complete!");
                onCompleteRef.current();
            } else {
                moveToNextPoint();
            }
        }, CAPTURE_DELAY);
    }, [moveToNextPoint, isCapturing, isProcessingSample, lastCapturedPosition, CAPTURE_DELAY]);

    useEffect(() => {
        if (!isActive || dotState !== 'active') {
            return;
        }
        
        if (samplesCollected >= maxSamples) {
            console.log("Max samples reached, not starting countdown");
            onComplete();
            return;
        }

        console.log("Starting countdown timer");

        const intervalId = setInterval(() => {
            setCountdown(prevCountdown => {
                const newCountDown = prevCountdown - (COUNTDOWN_INTERVAL / 1000);
                console.log(`Countdown tick: ${newCountDown.toFixed(1)}`);

                if (newCountDown <= 0) {
                    console.log("Countdown complete - capturing");
                    clearInterval(intervalId);
                    captureCurrentSample();
                    return 0;
                } else {
                    return newCountDown;
                }
            });
        }, COUNTDOWN_INTERVAL);

        return () => {
            console.log("Cleaning up countdown timer");
            clearInterval(intervalId);
        };
    }, [isActive, dotState, captureCurrentSample, COUNTDOWN_INTERVAL]);

    useEffect(() => {
        if (isActive) {
            console.log("Calibration activated");
            setCurrentGridIndex(0);
            const startPosition = getGridPosition(0);
            setCurrentDot(startPosition);
            setDotState('waiting');
            setCountdown(COUNTDOWN_TIME);
            setIsCapturing(false);
            setIsProcessingSample(false);
            setLastCapturedPosition(null);

            setTimeout(() => {
                if (samplesCollectedRef.current >= maxSamplesRef.current) {
                    console.log("Max samples already reached, not starting calibration");
                    onCompleteRef.current();
                    return;
                }

                console.log("Starting first dot sequence");
                setDotState('active');
                setCountdown(COUNTDOWN_TIME);
            }, DOT_TRANSITION_DELAY);
        } else {
            console.log("Calibration deactivated");
            setDotState('waiting');
            setCountdown(COUNTDOWN_TIME);
            setCurrentGridIndex(0);
            setIsCapturing(false);
            setIsProcessingSample(false);
            setLastCapturedPosition(null);
        }
    }, [isActive, getGridPosition, COUNTDOWN_TIME, DOT_TRANSITION_DELAY]);

    useEffect(() => {
        const handleKeyPress = (event) => {
            if (!isActive) return;

            if (samplesCollected >= maxSamples) {
                if (event.key === 'Escape') {
                    console.log("ESC pressed - max samples reached");
                    onCancel();
                }
                return;
            }

            if (event.key === 'Escape') {
                console.log("ESC pressed");
                onCancel();
            } else if (event.key === ' ' || event.key === 'Enter') {
                event.preventDefault();
                event.stopPropagation();

                console.log("Space/Enter pressed");
                if (dotState === 'waiting') {
                    setDotState('active');
                    setCountdown(COUNTDOWN_TIME);
                } else if (dotState === 'active' && !isProcessingSample) {
                    captureCurrentSample();
                }
            }
        };

        window.addEventListener('keydown', handleKeyPress);
        return () => window.removeEventListener('keydown', handleKeyPress);
    }, [isActive, dotState, onCancel, captureCurrentSample, isProcessingSample, COUNTDOWN_TIME]);

    if (!isActive) return null;

    const dotSize = 25;
    const dotX = currentDot.x * window.innerWidth - dotSize / 2;
    const dotY = currentDot.y * window.innerHeight - dotSize / 2;

    function getDotColor() {
        switch (dotState) {
            case 'waiting': 
                return '#ff4757'; // Red
            case 'active':
                return '#ff6348'; // Orange-red
            case 'capturing':
                return '#2ed573'; // Green
            default:
                return '#ff4757'; 
        }
    }

    function getDotShadow() {
        switch (dotState) {
            case 'waiting': return '0 0 20px rgba(255, 71, 87, 0.8), 0 0 40px rgba(255, 71, 87, 0.4)';
            case 'active': return '0 0 30px rgba(255, 99, 72, 1), 0 0 60px rgba(255, 99, 72, 0.6)';
            case 'capturing': return '0 0 25px rgba(46, 213, 115, 0.9), 0 0 50px rgba(46, 213, 115, 0.5)';
            default: return '0 0 20px rgba(255, 71, 87, 0.8)';
        }
    }

    function getDotTransform() {
        const scale = dotState === 'active' ? 1.3 : dotState === 'capturing' ? 0.9 : 1;
        const pulse = dotState === 'waiting' ? 'scale(1.1)' : '';
        return `scale(${scale}) ${pulse}`;
    }

    return (
        <>
            {/* Calibration dot */}
            <div 
                style={{
                    position: 'fixed',
                    left: `${dotX}px`,
                    top: `${dotY}px`,
                    width: `${dotSize}px`,
                    height: `${dotSize}px`,
                    borderRadius: '50%',
                    backgroundColor: getDotColor(),
                    boxShadow: getDotShadow(),
                    transform: getDotTransform(),
                    transition: 'all 0.2s ease',
                    border: '3px solid white',
                    zIndex: 1001,
                    pointerEvents: 'none'
                }}
            />

            {/* Countdown display */}
            {/*{dotState === 'active' && countdown > 0 && (
                <div style={{
                    position: 'fixed',
                    left: `${dotX + dotSize / 2}px`,
                    top: `${dotY - 35}px`,
                    transform: 'translateX(-50%)',
                    color: 'white',
                    fontSize: '20px',
                    fontWeight: 'bold',
                    fontFamily: 'Arial, sans-serif',
                    textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)',
                    zIndex: 1001,
                    pointerEvents: 'none'
                }}>
                    {countdown}
                </div>
            )}*/}

            {/* Progress indicator */}
            <div style={{
                position: 'fixed',
                top: '20px',
                right: '20px',
                backgroundColor: 'rgba(0, 0, 0, 0.7)',
                color: 'white',
                padding: '10px 15px',
                borderRadius: '8px',
                fontFamily: 'Arial, sans-serif',
                fontSize: '14px',
                zIndex: 1001,
                pointerEvents: 'none'
            }}>
                <div>Calibrating: {samplesCollected}/{maxSamples}</div>
                <div style={{
                    width: '100px',
                    height: '4px',
                    backgroundColor: 'rgba(255, 255, 255, 0.3)',
                    borderRadius: '2px',
                    marginTop: '5px',
                    overflow: 'hidden',
                }}>
                    <div style={{
                        width: `${progressPercent}%`,
                        height: '100%',
                        backgroundColor: '#4CAF50',
                        transition: 'width 0.3s ease'
                    }} />
                </div>
                <div style={{
                    fontSize: '12px',
                    marginTop: '3px',
                    color: '#ccc'
                }}>
                    ESC to cancel | SPACE to capture
                </div>
            </div>
        </>
    );
};

export default CalibrationDot;