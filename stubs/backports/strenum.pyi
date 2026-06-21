from enum import Enum

class StrEnum(str, Enum):
    """Backport of enum.StrEnum (Python < 3.11)."""
