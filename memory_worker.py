import json
import sys
from pathlib import Path

import numpy as np
import psutil

from runtime_config import create_cpu_session, session_configuration


WARMUP_RUNS = 100
MEASUREMENT_RUNS = 1000


def check_peak_memory(model_path):
    process = psutil.Process()
    rss_before_session_bytes = process.memory_info().rss
    session = create_cpu_session(model_path)
    input_name = session.get_inputs()[0].name

    input_data = np.zeros((1, 3, 32, 32), dtype=np.float32)
    input_feed = {input_name: input_data}

    for _ in range(WARMUP_RUNS):
        session.run(None, input_feed)

    for _ in range(MEASUREMENT_RUNS):
        session.run(None, input_feed)

    peak_working_set_bytes = process.memory_info().peak_wset
    return {
        "peak_working_set_bytes": peak_working_set_bytes,
        "peak_working_set_mib": peak_working_set_bytes / (1024 ** 2),
        "rss_before_session_bytes": rss_before_session_bytes,
        "batch_size": 1,
        "warmup_runs": WARMUP_RUNS,
        "measurement_runs": MEASUREMENT_RUNS,
        "scope": "worker process startup, session initialization, and inference",
        "input": {
            "source": "fixed zero array",
            "shape": list(input_data.shape),
            "dtype": str(input_data.dtype),
        },
        "runtime": session_configuration(session),
    }



def main():
    if len(sys.argv) != 2:
        raise SystemExit("사용법: python memory_worker.py MODEL_PATH")

    model_path = Path(sys.argv[1]).resolve(strict=True)
    print(json.dumps(check_peak_memory(model_path)))


if __name__ == "__main__":
    main()
