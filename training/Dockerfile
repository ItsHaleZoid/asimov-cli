# Use PyTorch base image with CUDA support and build tools
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install additional system dependencies (PyTorch image has most CUDA tools)
RUN apt-get update && apt-get install -y \
    git \
    openssh-client \
    wget \
    curl \
    gcc \
    g++ \
    make \
    build-essential \
    cmake \
    vim \
    htop \
    libblas3 \
    liblapack3 \
    liblapack-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    

# Set python3 as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create a working directory and copy necessary files.
WORKDIR /app
COPY requirements.txt .
COPY start_training.sh .

# Grant execution permissions to the start_training.sh script
RUN chmod +x start_training.sh

# Upgrade pip and install all Python packages from the requirements file in a single, efficient layer.
# This approach minimizes image size and build time by caching the dependencies.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt
