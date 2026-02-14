import subprocess
import time
from datetime import datetime
from typing import Tuple


def deploy_docling(
    model_name: str,
    target_message: str = "Application startup complete",
    max_dep_time: int = 2400,
    port: int = 8000,
    gpu_memory_utilization: float = 0.90,
    max_model_length: int = 6144,
    max_num_seqs: int = 1,
    max_num_batched_tokens: int = 1024,
    host: str = "0.0.0.0",
) -> Tuple[bool, str]:
    served_model_name = "_".join(model_name.split("/")[-2:])

    cmd1 = """pkill -f "vllm.entrypoints.openai.api_server .*--port {port}" || true""".format(
        **{
            "port": port,
        }
    )

    cmd2 = """export TOKENIZERS_PARALLELISM=true

vllm serve \
    --model {model_name} \
    --served-model-name {served_model_name} \
    --host {host} \
    --port {port} \
    --gpu-memory-utilization {gpu_memory_utilization} \
    --max-model-len {max_model_length} \
    --max-num-seqs {max_num_seqs} \
    --max-num-batched-tokens {max_num_batched_tokens} \
    --disable-log-stats \
    --revision untied \
    --enable-prefix-caching \
    --trust-remote-code > {served_model_name}_{port}.log 2>&1 &

sleep 3; tail -n 80 {served_model_name}_{port}.log""".format(
        **{
            "model_name": model_name,
            "served_model_name": served_model_name,
            "port": port,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_length": max_model_length,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "host": host,
        }
    )

    t0 = datetime.now()

    subprocess.run(cmd1, shell=True)

    time.sleep(3)

    subprocess.run(cmd2, shell=True)

    while True:
        time.sleep(5)

        with open(f"{served_model_name}_{port}.log", "r") as log_file:
            log_content = log_file.read()

        # Check if the target message is in the log content
        if target_message in log_content:
            break

        t1 = datetime.now()

        diff = t1 - t0
        if diff.seconds >= max_dep_time:
            return False, served_model_name, None

    return True, served_model_name, port
