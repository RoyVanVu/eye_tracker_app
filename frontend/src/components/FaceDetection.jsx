import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import { drawMesh } from "../utilities";
import axios from "axios"
import WebcamComponent from "./WebcamComponent";
import EyeCanvas from "./EyeCanvas";

function FaceDetection() {
    const webcamRef = useRef(null);
    const canvasRef = useRef(null);
    const leftEyeCanvasRef = useRef(null);
    const rightEyeCanvasRef = useRef(null);
    const [faceDetected, setFaceDetected] = useState(false);
    const [lastError, setLastError] = useState(null);

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

                    // console.log("Prediction conditions check:", {
                    //     isProcessing, 
                    //     serverStatus, 
                    //     leftBoxValid: leftBox.width > 0,
                    //     rightBoxValid: rightBox.width > 0,
                    //     leftBoxWidth: leftBox.width,
                    //     rightBoxWidth: rightBox.width
                    // });
                    
                    // if (!isProcessing && leftBox.width > 0 && rightBox.width > 0) {
                    //     console.log("All condition met - Calling predictGaze...");
                    //     predictGaze(leftEyeCanvasRef.current, rightEyeCanvasRef.current);
                    // } else {
                    //     console.log("Condition not met - skipping prediction");
                    // }
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
        </div>
    );
}

export default FaceDetection;