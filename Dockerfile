FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Create a non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} user && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash user

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

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt"

ARG DEV_MODE=false
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    if [ \"$DEV_MODE\" = \"true\" ] ; then pip install --no-cache-dir -r requirements-devtools.txt ; fi"

# Set ownership of /app to the user
RUN chown -R user:user /app

# Switch to non-root user
USER user

CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate base && /bin/bash"]
