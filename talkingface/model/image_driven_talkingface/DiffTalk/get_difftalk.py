from .inference import get_model

__all__ = ["get_difftalk_inference"]


def get_difftalk_inference():
    return get_model()
