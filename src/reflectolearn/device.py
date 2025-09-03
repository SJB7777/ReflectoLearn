import torch


def get_device_and_workers() -> tuple[torch.device, int]:
    if torch.backends.mps.is_available():
        return torch.device("mps"), 0
    elif torch.cuda.is_available():
        return torch.device("cuda"), 4
    else:
        return torch.device("cpu"), 0
