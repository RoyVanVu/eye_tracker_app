import React from "react";
import Webcam from "react-webcam";

const WebcamComponent = React.forwardRef((props, ref) => {
    return (
        <Webcam 
            ref={ref}
            style={{
                position: "absolute",
                marginLeft: "auto",
                marginRight: "auto",
                left: 0,
                right: 0,
                textAlign: "center",
                width: 640,
                height: 480,
                opacity: 0
            }}
            {...props}
        />
    );
});

WebcamComponent.displayName = "WebcamComponent";
export default WebcamComponent;