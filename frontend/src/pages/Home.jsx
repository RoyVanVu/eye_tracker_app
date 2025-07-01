import React, { useRef, useState, useEffect } from "react";
import FaceDetection from "../components/FaceDetection";

function Home() {
    return (
        <div style={{
            height: '100vh',
            width: '100vw',
            overflow: 'hidden',
            position: 'relative'
        }}>
            <header className="App-header">
                <FaceDetection />
            </header>
        </div>
    );
}

export default Home;