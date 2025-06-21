import React from "react";

const EyeCanvas = React.forwardRef(({ position, marginLeft, ...props }, ref) => {
    return (
        <canvas 
            ref={ref}
            width={224}
            height={224}
            style={{
                position: "absolute",
                marginLeft: marginLeft,
                marginRight: "auto",
                left: 0,
                right: 0,
                textAlign: "left",
                zIndex: 10,
                width: 150,
                height: 150,
                border: "2px solid green"
            }}
            {...props}
        />
    );
});

EyeCanvas.displayName = "EyeCanvas";
export default EyeCanvas;