curl <http://35.200.249.186:8080/v1/chat/completions> -H "Content-Type: application/json" -d '{"model": "bitnet","messages": [{"role": "user", "content": "Hello, how are you?"}],"temperature": 0.7 }'

    1  sudo apt update
    2  sudo apt upgrade
    4  sudo apt install clang
    5  `bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"`
    6  sudo `bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"`
    7  sudo su
    8  sudo `bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"`
    9  sudo bash -c "$(wget -O - <https://apt.llvm.org/llvm.sh>)"
   10  wget <https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh>
   11  bash Miniconda3-latest-Linux-x86_64.sh
   12  source ~/.bashrc
   13  conda --version
   15  sudo rm -r Miniconda3-latest-Linux-x86_64.sh
   17  git clone --recursive <https://github.com/Ravilochan/BitNet.git>
   18  cd BitNet/
   19  conda create -n bitnet-cpp python=3.9
   20  conda activate bitnet-cpp
   21  pip install -r requirements.txt
   22  # Manually download the model and run with local path
   23  huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
   24  python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
   28  export CC=clang
   29  export CXX=clang++
   32  sudo apt install cmake
   33  python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
   34  python run_server.py --host 0.0.0.0 --threads 8 --ctx-size 8192 --n-predict 8192 --mlock --parallel 8 --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf > server.log 2>&1 &
