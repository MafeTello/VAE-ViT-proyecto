# models/build.py
from .model_zoo import TimmRegressor

def build_model(config):
    model_type = config.MODEL.TYPE

    if model_type.startswith("baevit"):
        model_name = config.MODEL.NAME
        img_size = config.DATA.IMG_SIZE
        last_feature_dim = 576  # igual al último embedding

        model = TimmRegressor(
            model_name=model_name,
            feature_dim=last_feature_dim,
            is_sigmoid=False,
            img_size=img_size,
            config=config,
        )
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")

    return model
