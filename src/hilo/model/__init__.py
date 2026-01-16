from .hilo import HiLO
import logging


all_models = {
    "hilo": HiLO,
}


def build_model(cfg):
    logger = logging.getLogger(__name__)
    model_name = cfg["name"].lower()

    if model_name in all_models:
        model = all_models[model_name](cfg)
        logger.info(f"Built model {model_name}: \n{model}")
        return model

    raise NotImplementedError(f"Model '{model_name}' not implemented yet.")
