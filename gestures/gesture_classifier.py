from enum import Enum


class GestureType(str, Enum):
    UNKNOWN = "unknown"
    OPEN_PALM = "open_palm"
    FIST = "fist"


class GestureClassifier:
    def classify(self, finger_count: int) -> GestureType:
        if finger_count >= 4:
            return GestureType.OPEN_PALM
        if finger_count <= 1:
            return GestureType.FIST
        return GestureType.UNKNOWN
