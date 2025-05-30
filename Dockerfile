FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /uvx /bin/

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

COPY . /app
WORKDIR /app

# Create data and logs directories and set permissions
RUN mkdir -p /app/data /app/logs && \
    chown -R user:user /app/data /app/logs && \
    chmod 755 /app/data /app/logs

RUN conda init bash

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    uv pip install ."

ARG DEV_MODE=false
RUN if [ "$DEV_MODE" = "true" ] ; then \
    /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    uv pip install -e '.[dev]'" ; \
    fi

RUN chown -R user:user /app

USER user

CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate base && /bin/bash"]
