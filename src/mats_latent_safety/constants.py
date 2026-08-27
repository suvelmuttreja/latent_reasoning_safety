"""Protocol constants shared by training and evaluation."""

START_LATENT = "<|start-latent|>"
END_LATENT = "<|end-latent|>"
LATENT = "<|latent|>"
IGNORE_INDEX = -100


def k_for_stage(stage: int, c_thought: int = 2, max_stage: int = 3) -> int:
    """Return the registered number of latent positions for a stage."""
    if stage < 0:
        raise ValueError("stage must be non-negative")
    if c_thought < 0 or max_stage < 0:
        raise ValueError("c_thought and max_stage must be non-negative")
    return min(stage, max_stage) * c_thought


def optimizer_updates(
    examples: int,
    epochs: int,
    effective_batch_size: int,
) -> int:
    """Exact stage update count, including a final partial batch per epoch."""
    if examples <= 0 or epochs <= 0 or effective_batch_size <= 0:
        raise ValueError("examples, epochs, and effective_batch_size must be positive")
    return ((examples + effective_batch_size - 1) // effective_batch_size) * epochs

