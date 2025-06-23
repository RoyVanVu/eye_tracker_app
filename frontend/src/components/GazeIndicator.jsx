import React from 'react';

const GazeIndicator = ({ gazeCoordinates, serverStatus, isVisible = true }) => {
    if (!isVisible || 
        serverStatus !== "Connected" || 
        gazeCoordinates.x === undefined || 
        gazeCoordinates.x === 0) {
        return null;
    }

    return (
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
            pointerEvents: "none", 
        }} />
    );
};

export default GazeIndicator;