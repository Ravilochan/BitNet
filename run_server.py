#!/usr/bin/env python3

import os
import sys
import signal
import platform
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_server")

def run_command(command, shell=False):
    """Run a system command and ensure it succeeds."""
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred while running command: {e}")
        sys.exit(1)

def run_server():
    build_dir = "build"
    if platform.system() == "Windows":
        server_path = os.path.join(build_dir, "bin", "Release", "llama-server.exe")
        if not os.path.exists(server_path):
            server_path = os.path.join(build_dir, "bin", "llama-server")
    else:
        server_path = os.path.join(build_dir, "bin", "llama-server")
    
    if not os.path.exists(server_path):
        logger.error(f"Server binary not found at {server_path}. Make sure to run setup_env.py first.")
        sys.exit(1)
    
    # Create command with all necessary parameters
    command = [
        f'{server_path}',
        '-m', args.model,
        '-c', str(args.ctx_size),
        '--port', str(args.port),
        '--host', args.host,
        '-t', str(args.threads),
        '--batch-size', str(args.batch_size),
        '-ngl', '0',  # No GPU layers by default
        '-n', str(args.n_predict),
        '--temp', str(args.temperature),
    ]
    
    # Add optional parameters if specified
    if args.n_gpu_layers > 0:
        command[command.index('-ngl') + 1] = str(args.n_gpu_layers)
    
    # Add system prompt if provided
    if args.prompt:
        command.extend(['-p', args.prompt])
    
    # Add continuous batching by default for better performance
    if args.cont_batching:
        command.extend(['--cont-batching'])
    else:
        command.extend(['-cb'])  # Enable continuous batching by default
    
    # Add additional parameters for production
    if args.mlock:
        command.extend(['--mlock'])
    
    if args.numa:
        command.extend(['--numa'])
        
    if args.parallel > 0:
        command.extend(['--parallel', str(args.parallel)])
    
    if args.embedding:
        command.extend(['--embedding'])
        
    if args.api_key:
        command.extend(['--api-key', args.api_key])
    
    logger.info(f"Starting server with command: {' '.join(command)}")
    logger.info(f"OpenAI-compatible API will be available at: http://{args.host}:{args.port}/v1/chat/completions")
    logger.info(f"Web UI will be available at: http://{args.host}:{args.port}")
    
    run_command(command)

def signal_handler(sig, frame):
    logger.info("Ctrl+C pressed, shutting down server...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description='Run BitNet model as an OpenAI-compatible API server')
    parser.add_argument("-m", "--model", type=str, help="Path to model file", required=True)
    parser.add_argument("-c", "--ctx-size", type=int, help="Size of the prompt context", required=False, default=4096)
    parser.add_argument("-p", "--port", type=int, help="Server port", required=False, default=8080)
    parser.add_argument("--host", type=str, help="Server host", required=False, default="127.0.0.1")
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", required=False, default=4)
    parser.add_argument("-b", "--batch-size", type=int, help="Batch size for prompt processing", required=False, default=512)
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, help="Number of GPU layers to use", required=False, default=0)
    parser.add_argument("-n", "--n-predict", type=int, help="Number of tokens to predict (max output length)", required=False, default=4096)
    parser.add_argument("--temperature", type=float, help="Temperature for sampling (higher = more creative, lower = more deterministic)", required=False, default=0.8)
    parser.add_argument("--prompt", type=str, help="System prompt for the model", required=False)
    parser.add_argument("--mlock", action='store_true', help="Force system to keep model in RAM rather than swapping or compressing")
    parser.add_argument("--numa", action='store_true', help="Enable NUMA support for multiple CPU sockets")
    parser.add_argument("--parallel", type=int, help="Number of parallel sequences to decode", required=False, default=0)
    parser.add_argument("--cont-batching", action='store_true', help="Enable continuous batching for efficient multiple concurrent requests")
    parser.add_argument("--embedding", action='store_true', help="Enable embedding extraction endpoint")
    parser.add_argument("--api-key", type=str, help="API key for server authentication", required=False)

    args = parser.parse_args()
    
    # Ensure model file exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found at {args.model}. Please run setup_env.py first.")
        sys.exit(1)
    
    run_server()