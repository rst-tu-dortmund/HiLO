import torch


def move_data_to_device(batch, device):
    if isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = move_data_to_device(v, device)
    elif isinstance(batch, (list, tuple)):
        batch = [move_data_to_device(item, device) for item in batch]
    elif isinstance(batch, torch.Tensor):
        batch = batch.to(device)
    return batch
