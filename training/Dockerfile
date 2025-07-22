# Use a smaller runtime image as we don't need the full CUDA SDK
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and clean up apt cache in the same layer
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    

# Set python3 as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create a working directory. This is where the code will be cloned.
WORKDIR /app

# Copy the requirements file.
COPY requirements.txt .
COPY train.py .

# Upgrade pip and install packages from requirements.txt in a single layer
# This is more robust and creates a smaller image.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt
