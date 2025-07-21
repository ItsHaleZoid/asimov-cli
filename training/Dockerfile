# Use a smaller runtime image as we don't need the full CUDA SDK
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, including the SSH server
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create a working directory
WORKDIR /app

# Copy only the necessary files
COPY requirements.txt .
COPY train.py .

# Upgrade pip and install packages from requirements.txt in a single layer
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# This will be overridden by the --args flag, but is good practice to have.
ENTRYPOINT ["python", "/app/train.py"]
