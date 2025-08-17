FROM pytorch/pytorch

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pybase64 \
    numpy \
    image \
    opencv-python \
    traceback2 \
    Flask \
    flask-cors \
    requests 

RUN mkdir -p /app/model && chmod 777 /app/model
RUN mkdir -p /app/calibration && chmod 777 /app/calibration

VOLUME /app/model
VOLUME /app/calibration

COPY server.py /app/server.py
COPY test_backend.py /app/test_backend.py

CMD ["python", "server.py"]