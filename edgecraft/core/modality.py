from enum import Enum, auto

class Modality(str, Enum):
    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"
    MULTIMODAL = "multimodal"
    TIME_SERIES = "time_series"
    STRUCTURED = "structured"

class TaskType(str, Enum):
    # Vision
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    POSE_ESTIMATION = "pose_estimation"
    CROWD_COUNTING = "crowd_counting"

    # Audio
    SPEECH_RECOGNITION = "speech_recognition"
    AUDIO_CLASSIFICATION = "audio_classification"

    # Text
    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"

    # General
    ANOMALY_DETECTION = "anomaly_detection"
    REGRESSION = "regression"
