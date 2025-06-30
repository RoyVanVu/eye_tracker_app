import React, { useState, useEffect, useCallback } from 'react';

const CalibrationDot = ({
    isActive,
    onCaptureSample,
    onComplete,
    onCancel,
    samplesCollected = 0,
    maxSamples = 50,
    progressPercent = 0
}) => {
    const [currentDot, setCurrentDot] = useState({ 
        x: 0.5,
        y: 0.5
    });
    const [dotState, setDotState] = useState('waiting');
    const [countdown, setCountdown] = useState(3);

    const generateRandomPosition = useCallback(() => {
        const margin = 0.05;
        const x = margin + Math.random() * (1 - 2 * margin);
        const y = margin + Math.random() * (1 - 2 * margin);
        return { x, y };
    }, []);

    const moveToNextPoint = useCallback(() => {
        setCurrentDot(generateRandomPosition());
        setDotState('waiting');
        setCountdown(3);
    }, [generateRandomPosition]);

    const startDotSequence = useCallback(() => {
        setDotState('active');
        setCountdown(3);
    }, []);

    const captureSample = useCallback(() => {
        setDotState('capturing');
        onCaptureSample(currentDot.x, currentDot.y);
        setTimeout(() => {
            if (samplesCollected + 1 >= maxSamples) {
                onComplete();
            } else {
                moveToNextPoint();
            }
        }, 800);
    }, [currentDot, onCaptureSample, onComplete, samplesCollected, maxSamples, moveToNextPoint]);

    useEffect(() => {
        if (dotState === 'active' && countdown > 0) {
            const timer = setTimeout(() => {
                setCountdown(prev => prev - 1);
            }, 1000);
            return () => clearTimeout(timer);
        } else if (dotState === 'active' && countdown === 0) {
            captureSample();
        }
    }, [dotState, countdown, captureSample]);

    useEffect(() => {
        if (isActive) {
            const timer = setTimeout(() => {
                startDotSequence();
            }, 1000);
            return () => clearTimeout(timer);
        }
    }, [isActive, startDotSequence]);

    useEffect(() => {
        const handleKeyPress = (event) => {
            if (!isActive) return;

            if (event.key === 'Escape') {
                onCancel();
            } else if (event.key === ' ' || event.key === 'Enter') {
                if (dotState === 'waiting') {
                    startDotSequence();
                } else if (dotState === 'active' && countdown === 0) {
                    captureSample();
                }
            }
        };
        window.addEventListener('keydown', handleKeyPress);
        return () => window.removeEventListener('keydown', handleKeyPress);
    }, [isActive, dotState, countdown, onCancel, startDotSequence, captureSample]);

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
                    transition: 'all 0.5s ease',
                    border: '3px solid white',
                    zIndex: 1000,
                    pointerEvents: 'none'
                }}
            />

            {/* Countdown number above the dot */}
            {dotState === 'active' && countdown > 0 && (
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
            )}

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
                    ESC to cancel
                </div>
            </div>
        </>
    );
};

export default CalibrationDot;