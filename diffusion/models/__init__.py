from .rft_img import RFT


def get_model_cls(model_id):
    if model_id == "rft":
        return RFT
