import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def describe_device() -> dict:
    d = get_device()
    result = {"device": str(d), "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        result.update({
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })
    return result
