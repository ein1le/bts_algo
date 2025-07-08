# Multi-stage Docker build for BTS Algorithmic Trading
FROM nvidia/cuda:12.2-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    libssl-dev \
    libboost-all-dev \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TensorFlow C API
RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.16.1.tar.gz \
    && tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-2.16.1.tar.gz \
    && ldconfig /usr/local/lib \
    && rm libtensorflow-gpu-linux-x86_64-2.16.1.tar.gz

# Set TensorFlow environment variables
ENV TENSORFLOW_ROOT=/usr/local
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Copy source code
WORKDIR /app
COPY . .

# Build C++ components
RUN mkdir -p build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

# Runtime stage
FROM nvidia/cuda:12.2-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libboost-system1.74.0 \
    libssl3 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy TensorFlow C library from builder
COPY --from=builder /usr/local/lib/libtensorflow* /usr/local/lib/
COPY --from=builder /usr/local/include/tensorflow /usr/local/include/tensorflow
RUN ldconfig /usr/local/lib

# Create app user
RUN useradd -m -u 1000 appuser
USER appuser
WORKDIR /home/appuser/app

# Copy built executable and Python code
COPY --from=builder --chown=appuser:appuser /app/build/cpp/bts_algo ./bin/
COPY --from=builder --chown=appuser:appuser /app/python ./python/
COPY --from=builder --chown=appuser:appuser /app/requirements.txt ./

# Install Python dependencies
USER root
RUN pip3 install --no-cache-dir -r requirements.txt
USER appuser

# Set environment variables
ENV PATH="/home/appuser/app/bin:$PATH"
ENV PYTHONPATH="/home/appuser/app/python:$PYTHONPATH"

# Expose ports (if needed for monitoring)
EXPOSE 8080

# Default command
CMD ["./bin/bts_algo"] 