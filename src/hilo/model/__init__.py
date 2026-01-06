from .hilo import HiLO
from.legacy.model import Transformer as LegacyHiLO
import logging


all_models = {
    "hilo": HiLO,
    "legacy_hilo": LegacyHiLO,
}


def build_model(cfg):
    logger = logging.getLogger(__name__)
    model_name = cfg["name"].lower()

    if model_name in all_models:
        model = all_models[model_name](cfg)
        logger.info(f"Built model {model_name}: \n{model}")
        return model

    raise NotImplementedError(f"Model '{model_name}' not implemented yet.")
