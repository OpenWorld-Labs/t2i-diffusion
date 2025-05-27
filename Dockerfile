FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y bash \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    ffmpeg \
    python3.10 \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create directories for bind mounts and set ownership
RUN mkdir -p /app/data /app/logs && \
    chmod 777 /app/data /app/logs && \
    chown -R root:root /app

COPY requirements.txt .

# Initialize conda for shell interaction
RUN conda init bash

# Add build argument for development mode
ARG DEV_MODE=false
COPY requirements-devtools.txt .

# Install requirements in the conda environment
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    if [ \"$DEV_MODE\" = \"true\" ] ; then pip install --no-cache-dir -r requirements-devtools.txt ; fi"

# Set the default command to source conda and start bash
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate base && /bin/bash"]
