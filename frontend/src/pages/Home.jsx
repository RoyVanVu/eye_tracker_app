import React, { useRef, useState, useEffect } from "react";
import FaceDetection from "../components/FaceDetection";

function Home() {
    return (
        <div>
            <header className="App-header">
                <FaceDetection />
            </header>
        </div>
    );
}

export default Home;