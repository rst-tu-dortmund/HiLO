from .fusion import FusionFilteredDataset
from torch.utils.data import DataLoader
import logging

all_datasets = {
    "fusion_filtered": FusionFilteredDataset,
}


def build_dataloader(cfg, split="train"):
    dataset_class = all_datasets[cfg["name"]]
    dataset = dataset_class(cfg, split=split)

    logger = logging.getLogger(__name__)
    logger.info(f"Built dataset {str(dataset)}")

    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=cfg.get("shuffle", split == "train"),
        num_workers=cfg.get("num_workers", 0),
        collate_fn=dataset.collate_fn,
        drop_last=cfg.get("drop_last", False),
    )
