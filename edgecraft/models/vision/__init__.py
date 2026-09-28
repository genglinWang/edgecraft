"""Vision model definitions for EdgeCraft."""
from edgecraft.models.vision.ultralytics import register_all as register_ultralytics
from edgecraft.models.vision.timm_models import register_all as register_timm


def register_all() -> None:
    """Register all vision models."""
    register_ultralytics()
    register_timm()
