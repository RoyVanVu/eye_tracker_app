import React from 'react';

const DebugInfo = ({
    serverStatus, 
    faceDetected, 
    isProcessing, 
    predictionCount, 
    gazeCoordinates, 
    lastError 
}) => {
    return (
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
    );
};

export default DebugInfo;