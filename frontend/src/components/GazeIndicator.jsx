import React from 'react';

const GazeIndicator = ({ gazeCoordinates, serverStatus, isVisible = true }) => {
    if (!isVisible || 
        serverStatus !== "Connected" || 
        gazeCoordinates.x === undefined || 
        gazeCoordinates.x === 0) {
        return null;
    }

    const rawX = gazeCoordinates.x * window.innerWidth;
    const rawY = gazeCoordinates.y * window.innerHeight;

    const padding = 5;
    const indicatorSize = 20;
    const halfSize = indicatorSize / 2;

    const clampedX = Math.min(
        Math.max(rawX, padding + halfSize),
        window.innerWidth - padding - halfSize
    );
    const clampedY = Math.min(
        Math.max(rawY, padding + halfSize),
        window.innerHeight - padding - halfSize
    );

    return (
        <div style={{
            position: "fixed",
            left: `${clampedX}px`,
            top: `${clampedY}px`,
            width: `${indicatorSize}px`,
            height: `${indicatorSize}px`,
            borderRadius: "50%",
            backgroundColor: "red",
            transform: "translate(-50%, -50%)",
            zIndex: 15,
            boxShadow: "0 0 10px rgba(255, 0, 0, 0.8)",
            pointerEvents: "none", 
        }} />
    );
};

export default GazeIndicator;