from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from heretic.schemas import EvaluationContext, MetricResult
from heretic.plugin import Plugin

if TYPE_CHECKING:
    from .config import Settings as HereticSettings
    from .model import Model


class Scorer(Plugin, ABC):
    """
    Abstract base class for scorer plugins.

    Scorers evaluate model behavior and return a MetricResult.
    
    Example: counting refusals, measuring KL divergence, etc.
    """

    class Settings(Plugin.Settings):
        """Scorer-specific settings with optimizer configuration."""
        use_in_optimizer: bool = Field(
            default=True,
            description="If true, this scorer's value is used as an Optuna objective.",
        )
        direction: Literal["minimize", "maximize"] = Field(
            default="minimize",
            description="Whether Optuna should minimize or maximize this scorer's value.",
        )
        label: str | None = Field(
            default=None,
            description="Optional display label for this scorer/metric.",
        )

    def __init__(
        self,
        settings: "HereticSettings",
        model: "Model",
        plugin_settings: BaseModel | None = None,
    ):
        super().__init__(plugin_settings=plugin_settings)
        self.settings = settings
        self.model = model

    def evaluate(self, ctx: EvaluationContext) -> MetricResult:
        """
        Evaluate this scorer given the evaluation context.
        
        Override this method in subclasses. Use `self.make_result()` to build
        the return value with settings-derived defaults.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement evaluate()"
        )

    def make_result(self, value: float, display: str | None = None) -> MetricResult:
        """
        Helper to build MetricResult with settings-derived defaults.
        
        Args:
            value: The numeric metric value.
            display: Human-readable string. Defaults to str(value).
        
        Returns:
            MetricResult with name/direction/use_in_optimizer from plugin_settings.
        """
        ps = self.plugin_settings
        return MetricResult(
            name=getattr(ps, "label", None) or self.name if ps else self.name,
            value=value,
            display=display if display is not None else str(value),
            use_in_optimizer=getattr(ps, "use_in_optimizer", True) if ps else True,
            direction=getattr(ps, "direction", "minimize") if ps else "minimize",
        )

    @staticmethod
    def required_response_metadata_fields() -> set[str]:
        """
        Response-level metadata fields needed by this scorer.
        Override to request fields like 'response_text', 'token_logprobs', etc.
        """
        return set()

    @staticmethod
    def required_context_metadata_fields() -> set[str]:
        """
        Context-level metadata fields needed by this scorer.
        Override to request fields like 'good_residuals', 'bad_residuals', etc.
        """
        return set()
