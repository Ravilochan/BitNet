FROM ubuntu:22.04

# avoid prompts, install system deps
ENV DEBIAN_FRONTEND=noninteractive
ENV CC=clang
ENV CXX=clang++
ENV CONDA_DIR=/opt/conda

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git clang cmake build-essential ca-certificates &&
    rm -rf /var/lib/apt/lists/*

# Install Miniconda heedlessly
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh &&
    bash /tmp/miniconda.sh -b -p $CONDA_DIR &&
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create and activate conda env, install python and pip tools
RUN conda create -n bitnet-cpp python=3.9 -y &&
    conda clean -afy

# Use bash login shell so `conda activate` works
SHELL ["bash", "-lc"]

# Clone and install Python requirements
RUN conda activate bitnet-cpp &&
    git clone --recursive https://github.com/Ravilochan/BitNet.git /opt/bitnet &&
    pip install --upgrade pip &&
    pip install -r /opt/bitnet/requirements.txt

WORKDIR /opt/bitnet

# Download the gguf model into models/…
RUN mkdir -p models/BitNet-b1.58-2B-4T &&
    huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T

# Prepare environment and build C++ core
RUN conda activate bitnet-cpp && python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# Expose the default port (adjust if needed)
EXPOSE 8080

# When container starts, activate env and run server on all CPUs
ENTRYPOINT ["bash","-lc","\
  conda activate bitnet-cpp && \
  python run_server.py \
    --host 0.0.0.0 \
    --threads $(nproc) \
    --ctx-size 8192 \
    --n-predict 8192 \
    --mlock \
    --parallel $(nproc) \
    --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
"]
