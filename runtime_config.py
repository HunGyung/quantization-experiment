"""Shared ONNX Runtime CPU settings for every benchmarked model."""

from pathlib import Path

import onnxruntime as ort


PROVIDER = "CPUExecutionProvider"
INTRA_OP_NUM_THREADS = 6
INTER_OP_NUM_THREADS = 1
GRAPH_OPTIMIZATION_LEVEL = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
EXECUTION_MODE = ort.ExecutionMode.ORT_SEQUENTIAL


def create_cpu_session(model_path: str | Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = INTRA_OP_NUM_THREADS
    options.inter_op_num_threads = INTER_OP_NUM_THREADS
    options.graph_optimization_level = GRAPH_OPTIMIZATION_LEVEL
    options.execution_mode = EXECUTION_MODE
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=[PROVIDER]
    )


def session_configuration(session: ort.InferenceSession) -> dict:
    """Record the settings actually attached to this session."""
    options = session.get_session_options()
    return {
        "providers": session.get_providers(),
        "intra_op_num_threads": options.intra_op_num_threads,
        "inter_op_num_threads": options.inter_op_num_threads,
        "graph_optimization_level": str(options.graph_optimization_level),
        "execution_mode": str(options.execution_mode),
        "enable_mem_pattern": options.enable_mem_pattern,
        "enable_cpu_mem_arena": options.enable_cpu_mem_arena,
    }
