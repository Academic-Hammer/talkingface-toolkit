from enum import Enum

class EvaluatorType(Enum):
    """Type for evaluation metrics.

    - ``SYNC``: SYNC metrics like LSE-C, LSE-D, etc.
    - ``VIDEOQ``: Video quality metrics like FID, etc.
    - ``AUDIOQ``: Audio quality metrics like PESQ, etc.
    """

    SYNC = 1
    VIDEOQ = 2
    AUDIOQ = 3