"""
Exceptions Module

Contains custom exceptions for the Test-Time-Diffusion framework.
"""

class TestTimeDiffusionError(Exception):
    """Base exception for all framework errors."""
    pass

class RetrievalError(TestTimeDiffusionError):
    """Raised when document retrieval fails."""
    pass

class LLMGenerationError(TestTimeDiffusionError):
    """Raised when LLM generation fails or returns malformed data."""
    pass

class EvaluationError(TestTimeDiffusionError):
    """Raised during evaluation errors."""
    pass
