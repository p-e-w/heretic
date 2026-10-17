# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, TypeAlias, cast

import torch
import torch.nn.functional as F
from optuna import Trial
from peft import PeftModel
from peft.tuners.lora.layer import Linear
from peft.utils.integrations import dequantize_module_weight
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)
from torch import Tensor
from torch.optim import LBFGS

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.model import Model, ModuleIO
from heretic.modifier import Context, Modifier, Serializable
from heretic.streaming import LayerStream
from heretic.utils import Prompt, is_hf_path, print

ProjectionKey: TypeAlias = tuple[int, str, int]


@dataclass
class Parameters(Serializable):
    start_layer_index: int
    end_layer_index: int
    retention_weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_presentation_dict(self) -> dict[str, str]:
        return {
            name: f"{value:.4f}" if isinstance(value, float) else str(value)
            for name, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Parameters":
        return cls(**data)


class Settings(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    original_model: str = Field(
        min_length=1,
        description="Original checkpoint to restore toward on good prompts.",
    )
    original_model_commit: str | None = Field(
        default=None,
        description="Commit hash of the original model on Hugging Face.",
    )
    good_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
        ),
        description="Prompts on which to restore the original model's behavior.",
    )
    retention_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
        ),
        description="Prompts on which to retain the starting edited model's behavior.",
    )
    streaming: bool = Field(
        default=False,
        description="Refresh student inputs after each edit while keeping both teachers frozen.",
    )
    lora_rank: PositiveInt = Field(
        default=64, description="Rank of the restoration adapter."
    )
    n_optimization_steps: PositiveInt = Field(
        default=5, description="Outer L-BFGS steps per projection."
    )
    learning_rate: PositiveFloat = Field(
        default=1.0, description="L-BFGS learning rate."
    )
    max_iter: PositiveInt = Field(
        default=20, description="Inner L-BFGS iterations per step."
    )
    history_size: PositiveInt = Field(default=10, description="L-BFGS history size.")
    update_penalty: NonNegativeFloat = Field(
        default=1e-4,
        description="Penalty on squared update norm relative to the starting weight norm.",
    )
    print_loss: bool = Field(
        default=False, description="Print the loss after each L-BFGS step."
    )


def flatten_io(module_io: ModuleIO) -> dict[ProjectionKey, tuple[Tensor, Tensor]]:
    return {
        (layer_index, component, module_index): pair
        for layer_index, layer_io in enumerate(module_io)
        for component, modules in layer_io.items()
        for module_index, pair in modules.items()
    }


@torch.no_grad()
def collect_frozen_projection_io(
    model: Model, prompts: list[Prompt]
) -> dict[ProjectionKey, tuple[Tensor, Tensor]]:
    """Evaluate frozen teacher projections in the same precision as the fit."""
    if isinstance(model.model, PeftModel) or getattr(
        model.model, "_hf_peft_config_loaded", False
    ):
        raise ValueError(
            "Frozen restoration targets require merged teacher checkpoints"
        )
    result = flatten_io(model.get_module_io_batched(prompts))
    for key, (inputs, _) in result.items():
        layer, component, index = key
        module = model.get_layer_modules(layer)[component][index]
        weight = dequantize_module_weight(module).detach().float()
        bias = getattr(module, "bias", None)
        if bias is not None:
            bias = cast(Tensor, bias).float().to(weight.device)
        # These inputs come from a full frozen-teacher run. Recomputing only the
        # projection in FP32 avoids fitting its BF16 rounding error as a repair.
        outputs = F.linear(inputs.float().to(weight.device), weight, bias)
        result[key] = (inputs, outputs.cpu())
    return result


def restoration_loss(
    prediction: Tensor,
    targets: Tensor,
    good_count: int,
    scale: Tensor,
    retention_weight: float,
) -> Tensor:
    """Balance population means so dataset sizes do not change their weights."""
    if prediction.shape != targets.shape or not 0 < good_count < len(targets):
        raise ValueError(
            "Restoration requires aligned, nonempty good and retention populations"
        )
    errors = (prediction - targets).square().mean(dim=1)
    return (
        errors[:good_count].mean() + retention_weight * errors[good_count:].mean()
    ) / scale


def fit_projection(
    module: Linear,
    inputs: Tensor,
    targets: Tensor,
    good_count: int,
    scale: Tensor,
    retention_weight: float,
    settings: Settings,
    generator: torch.Generator,
    label: str,
) -> None:
    """Fit a correction to the edited base without changing either teacher."""
    A = cast(Tensor, module.lora_A["default"].weight)
    B = cast(Tensor, module.lora_B["default"].weight)
    W = dequantize_module_weight(module.base_layer).detach().float().to(A.device)
    inputs = inputs.float().to(A.device)
    targets = targets.float().to(A.device)
    scale = scale.to(A.device)
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("Restoration requires a finite positive frozen scale")
    initial_A = torch.empty(A.shape)
    torch.nn.init.kaiming_uniform_(initial_A, a=math.sqrt(5), generator=generator)
    with torch.no_grad():
        A.copy_(initial_A)
        B.zero_()
        bias = module.base_layer.bias
        if bias is not None:
            bias = cast(Tensor, bias).float().to(A.device)
        base_output = F.linear(inputs, W, bias)
        weight_squared = W.square().sum().clamp_min(1.0)

    optimizer = LBFGS(
        [A, B],
        lr=settings.learning_rate,
        max_iter=settings.max_iter,
        history_size=settings.history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        prediction = base_output + (inputs @ A.T) @ B.T
        loss = restoration_loss(
            prediction, targets, good_count, scale, retention_weight
        )
        # The penalty depends on the update, not the arbitrary scale of its factors.
        update_squared = ((A @ A.T) * (B.T @ B)).sum().clamp_min(0)
        loss = loss + settings.update_penalty * update_squared / weight_squared
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite restoration loss")
        loss.backward()
        if (
            A.grad is None
            or B.grad is None
            or not torch.isfinite(A.grad).all()
            or not torch.isfinite(B.grad).all()
        ):
            raise FloatingPointError("Nonfinite restoration gradient")
        return loss

    try:
        for step in range(settings.n_optimization_steps):
            loss = optimizer.step(closure)
            if settings.print_loss:
                print(f"\\[{label}] Step: {step + 1}, Loss: {loss.item():.6f}")
        if not torch.isfinite(A).all() or not torch.isfinite(B).all():
            raise FloatingPointError("Nonfinite restoration adapter")
    finally:
        optimizer.zero_grad(set_to_none=True)


class BackwardRestoration(Modifier[Parameters]):
    settings: Settings

    @property
    def modifier_name(self) -> str:
        return "Backward restoration"

    @property
    def reproducible(self) -> bool:
        # The main reproduction settings already pin the student and datasets.
        # A second teacher must also be pinned before offering reproduction.
        return bool(
            is_hf_path(self.settings.original_model)
            and re.fullmatch(
                r"[0-9a-fA-F]{40}", self.settings.original_model_commit or ""
            )
        )

    def _collect_original_outputs(
        self,
        model: Model,
        prompts: list[Prompt],
    ) -> dict[ProjectionKey, Tensor]:
        with model.reference_model(
            self.settings.original_model,
            self.settings.original_model_commit,
        ) as teacher:
            return {
                key: outputs
                for key, (_, outputs) in collect_frozen_projection_io(
                    teacher, prompts
                ).items()
            }

    def init(self, ctx: Context) -> None:
        model = ctx.get_model()
        if isinstance(model.model, PeftModel) or getattr(
            model.model, "_hf_peft_config_loaded", False
        ):
            raise ValueError(
                "Backward restoration requires a merged edited checkpoint, not an active adapter"
            )
        self.good_prompts = ctx.load_prompts(self.settings.good_prompts)
        self.retention_prompts = ctx.load_prompts(self.settings.retention_prompts)
        if not self.good_prompts or not self.retention_prompts:
            raise ValueError(
                "Backward restoration requires nonempty good and retention datasets"
            )
        self.good_count = len(self.good_prompts)
        self.prompts = self.good_prompts + self.retention_prompts

        print()
        print("Collecting frozen edited-model outputs...")
        student_io = collect_frozen_projection_io(model, self.prompts)
        expected_keys = {
            (layer, component, index)
            for layer in range(len(model.get_layers()))
            for component, modules in model.get_layer_modules(layer).items()
            for index in range(len(modules))
        }
        if set(student_io) != expected_keys or any(
            len(pair[0]) != len(self.prompts) for pair in student_io.values()
        ):
            raise ValueError(
                "Backward restoration requires aligned dense projection inputs"
            )

        print("Collecting frozen original-model outputs...")
        original_outputs = self._collect_original_outputs(model, self.good_prompts)
        if set(original_outputs) != expected_keys:
            raise ValueError(
                "The original and edited models must have matching projection layouts"
            )
        self.targets: dict[ProjectionKey, Tensor] = {}
        self.scales: dict[ProjectionKey, Tensor] = {}
        self.inputs: dict[ProjectionKey, Tensor] = {}
        for key, (inputs, outputs) in student_io.items():
            original = original_outputs[key]
            student_good = outputs[: self.good_count]
            if original.shape != student_good.shape:
                raise ValueError(
                    "The original and edited models must have matching projection outputs"
                )
            self.targets[key] = torch.cat((original, outputs[self.good_count :]))
            original = original.float()
            discrepancy = (
                (student_good.float() - original).square().mean(dim=1).median()
            )
            variance = (original - original.mean(dim=0)).square().mean()
            self.scales[key] = torch.maximum(discrepancy, 0.01 * variance).clamp_min(
                1e-12
            )
            if not self.settings.streaming:
                self.inputs[key] = inputs

        model.apply_lora(self.settings.lora_rank)

    def suggest_parameters(self, ctx: Context, trial: Trial) -> Parameters:
        layer_count = len(ctx.get_model().get_layers())
        return Parameters(
            start_layer_index=trial.suggest_int(
                "start_layer_index", 0, layer_count // 2
            ),
            end_layer_index=trial.suggest_int(
                "end_layer_index", layer_count // 2 + 1, layer_count
            ),
            retention_weight=trial.suggest_float(
                "retention_weight", 0.1, 100.0, log=True
            ),
        )

    def modify_model(self, ctx: Context, parameters: Parameters) -> None:
        model = ctx.get_model()
        if (
            not 0
            <= parameters.start_layer_index
            < parameters.end_layer_index
            <= len(model.get_layers())
        ):
            raise ValueError("Invalid restoration layer range")
        if (
            not math.isfinite(parameters.retention_weight)
            or parameters.retention_weight <= 0
        ):
            raise ValueError("The retention weight must be finite and positive")
        generator = torch.Generator().manual_seed(cast(int, self.heretic_settings.seed))
        stream = LayerStream(model, self.prompts) if self.settings.streaming else None
        if stream is not None:
            for _ in range(parameters.start_layer_index):
                stream.advance()
        for layer in range(parameters.start_layer_index, parameters.end_layer_index):
            for component, modules in model.get_layer_modules(layer).items():
                for index, module in enumerate(modules):
                    key = (layer, component, index)
                    inputs = (
                        self.inputs[key]
                        if stream is None
                        else stream.collect(module)[0]
                    )
                    fit_projection(
                        cast(Linear, module),
                        inputs,
                        self.targets[key],
                        self.good_count,
                        self.scales[key],
                        parameters.retention_weight,
                        self.settings,
                        generator,
                        f"{layer}/{component}/{index}",
                    )
            if stream is not None:
                stream.advance()

    def reset_model(self, ctx: Context) -> None:
        model = ctx.get_model()
        if not model.reset_model():
            model.apply_lora(self.settings.lora_rank)
