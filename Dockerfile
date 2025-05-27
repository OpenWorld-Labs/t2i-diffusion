FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y bash \
    build-essential \
    git \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY requirements-devtools.txt .

RUN conda init bash

ARG DEV_MODE=false

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    if [ \"$DEV_MODE\" = \"true\" ] ; then pip install --no-cache-dir -r requirements-devtools.txt ; fi"

CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate base && /bin/bash"]
