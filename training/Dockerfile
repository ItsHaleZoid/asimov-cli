# Use PyTorch base image with CUDA support and build tools
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install additional system dependencies, including the SSH server
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    openssh-server \
    wget \
    curl \
    gcc \
    g++ \
    make \
    build-essential \
    cmake \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Configure SSH for root login
RUN echo "root:vastai" | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    mkdir -p /run/sshd

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set python3 as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create a working directory and copy necessary files.
WORKDIR /app
COPY requirements.txt .
COPY start.sh .

# Grant execution permissions to the start script
RUN chmod +x start.sh

# Upgrade pip and install all Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Default envs to reduce noisy logs and avoid xet-based downloads
ENV RUST_LOG=error \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

# Expose SSH port
EXPOSE 22

# Start the SSH server and any other services
CMD ["/app/start.sh"]
