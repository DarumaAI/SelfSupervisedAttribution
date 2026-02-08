import subprocess
import sys
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    :param config_path: Path to the YAML configuration file.
    :type config_path: str
    :return: Configuration dictionary.
    :rtype: Dict[str, Any]
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)


def deploy_server(config_path: str):
    """
    Deploy the VLLM model in server mode using settings from the configuration file.

    :param config_path: Path to the configuration file.
    :type config_path: str
    """
    config = load_config(config_path)

    # Extract arguments from config
    model = config.get("model")
    if not model:
        print("Error: 'model' field is required in configuration.")
        sys.exit(1)

    tensor_parallel_size = str(config.get("tensor_parallel_size", 1))
    gpu_memory_utilization = str(config.get("gpu_memory_utilization", 0.90))
    host = config.get("host", "0.0.0.0")
    port = str(config.get("port", 8000))
    dtype = config.get("dtype", "auto")

    # Construct the command to run the VLLM OpenAI API server
    # We use sys.executable to ensure we use the same python interpreter
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--tensor-parallel-size",
        tensor_parallel_size,
        "--gpu-memory-utilization",
        gpu_memory_utilization,
        "--host",
        host,
        "--port",
        port,
        "--dtype",
        dtype,
    ]

    # Add any other optional arguments if present in config
    if "max_model_len" in config:
        cmd.extend(["--max-model-len", str(config["max_model_len"])])

    print(f"Starting VLLM server with command: {' '.join(cmd)}")

    try:
        # Run the server
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"Server creation failed with error: {e}")
        sys.exit(1)
