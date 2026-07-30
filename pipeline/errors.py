"""
Shared exception type for pipeline failures.
Every step raises PipelineError on failure so the orchestrator can
report exactly which step broke and why, in plain language back to Telegram.
"""


class PipelineError(Exception):
    """Raised by any pipeline step when it cannot continue."""

    def __init__(self, step: str, reason: str) -> None:
        self.step = step
        self.reason = reason
        super().__init__(f"[{step}] {reason}")
