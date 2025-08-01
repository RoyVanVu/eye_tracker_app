import React, { useRef, useState, useEffect } from "react";
import FaceDetection from "../components/FaceDetection";

function Home() {
    const handleClick = (event) => {
        const x = event.clientX;
        const y = event.clientY;
        alert(`Clicked at coordinates: X=${x}, Y=${y}`);
    };

    return (
        <div style={{
            height: '100vh',
            width: '100vw',
            overflow: 'hidden',
            position: 'relative'
        }}
        onClick={handleClick}
        >
            <header className="App-header">
                <FaceDetection />
            </header>
        </div>
    );
}

export default Home;