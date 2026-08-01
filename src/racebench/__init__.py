"""Small, dependency-free tools for inference-race experiments."""

from .score import ScorePolicy, effective_request_score

__all__ = ["ScorePolicy", "effective_request_score"]
__version__ = "0.1.0"
