# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Arbitrary-Rank Ablation (ARA) (Weidmann 2026)
# See https://github.com/p-e-w/heretic/pull/211 for more information.

import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import bitsandbytes.functional as BNB_F
import torch
import torch.linalg as LA
import torch.nn.functional as F
from optuna import Trial
from peft.tuners.lora.layer import Linear
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
)
from torch import Tensor
from torch.optim import LBFGS

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.modifier import Context, Modifier, Serializable
from heretic.utils import format_dataset_specification, print


@dataclass
class Parameters(Serializable):
    start_layer_index: int
    end_layer_index: int
    preserve_good_behavior_weight: float
    steer_bad_behavior_weight: float
    overcorrect_relative_weight: float
    neighbor_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_presentation_dict(self) -> dict[str, str]:
        return {
            name: (f"{value:.4f}" if isinstance(value, float) else f"{value}")
            for name, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable":
        return Parameters(**data)


class Settings(BaseModel):
    good_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
        ),
        description="Dataset of prompts that tend to produce desirable responses.",
    )

    bad_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
        ),
        description="Dataset of prompts that tend to produce undesirable responses.",
    )

    preserve_row_magnitudes: bool = Field(
        default=True,
        description=(
            "Whether to renormalize the rows of the modified matrices to preserve "
            "the original matrices' row magnitudes. This is believed to improve "
            'intelligence retention (see Lai 2025, "Magnitude-Preserving Orthogonal Ablation").'
        ),
    )

    lora_rank: PositiveInt = Field(
        default=50,
        description=(
            "The rank of the LoRA adapter to use. "
            'While mathematically, ARA is of "arbitrary" rank, experiments have shown that '
            "singular values tend to drop rapidly after a few dozen dimensions, and approximating "
            "the full transformation with a LoRA has many practical advantages."
        ),
    )

    n_optimization_steps: PositiveInt = Field(
        default=5,
        description="Number of (outer) L-BFGS optimization steps to perform.",
    )

    learning_rate: float = Field(
        default=1.0,
        description="Learning rate to use in the L-BFGS optimizer.",
    )

    max_iter: PositiveInt = Field(
        default=20,
        description="Maximum number of (inner) iterations to perform per (outer) L-BFGS optimization step.",
    )

    history_size: PositiveInt = Field(
        default=10,
        description="Number of past updates to store for approximating the Hessian matrix in the L-BFGS optimizer.",
    )

    print_loss: bool = Field(
        default=False,
        description="Whether to print the loss value for each L-BFGS optimization step.",
    )


# For each vector in the 2D-tensor `a`, computes the mean Euclidean distance
# to the `k` nearest neighbors of the vector among the vectors in the 2D-tensor `b`.
def mean_distances_to_knn(a: Tensor, b: Tensor, k: int) -> Tensor:
    distances = torch.cdist(a, b)
    nearest_distances, _ = distances.topk(k, dim=1, largest=False)
    return nearest_distances.mean(1)


# The objective function at the heart of ARA.
def ara_loss(
    good_output: Tensor,
    bad_output: Tensor,
    new_good_output: Tensor,
    new_bad_output: Tensor,
    parameters: Parameters,
) -> Tensor:
    # The outputs for "good" prompts should change as little as possible.
    preserve_good_behavior = ((new_good_output - good_output) ** 2).mean()

    steer_bad_behavior = (
        # Pull the outputs for "bad" prompts towards
        # the original outputs for "good" prompts.
        mean_distances_to_knn(
            new_bad_output,
            good_output,
            parameters.neighbor_count,
        ).mean()
        # Push the outputs for "bad" prompts away from
        # the original outputs for "bad" prompts.
        # In combination with the above, this overcorrects
        # away from the original residuals, which results
        # in stronger steering that can overcome more complex
        # refusal mechanisms.
        + parameters.overcorrect_relative_weight
        * -mean_distances_to_knn(
            new_bad_output,
            bad_output,
            parameters.neighbor_count,
        ).mean()
    )

    return (
        parameters.preserve_good_behavior_weight * preserve_good_behavior
        + parameters.steer_bad_behavior_weight * steer_bad_behavior
    )


class ARA(Modifier[Parameters]):
    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def modifier_name(self) -> str:
        return "Arbitrary-Rank Ablation (ARA)"

    def init(self, ctx: Context) -> None:
        model = ctx.get_model()

        print()
        print(
            f"Loading good prompts from [bold]{format_dataset_specification(self.settings.good_prompts)}[/]..."
        )
        good_prompts = ctx.load_prompts(self.settings.good_prompts)
        print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

        print()
        print(
            f"Loading bad prompts from [bold]{format_dataset_specification(self.settings.bad_prompts)}[/]..."
        )
        bad_prompts = ctx.load_prompts(self.settings.bad_prompts)
        print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

        print()
        print("Obtaining module I/O for good prompts...")
        self.good_module_io = model.get_module_io_batched(good_prompts)

        print()
        print("Obtaining module I/O for bad prompts...")
        self.bad_module_io = model.get_module_io_batched(bad_prompts)

        # LoRA B matrices are initialized to zero by default in PEFT,
        # so we don't need to do anything manually.
        model.apply_lora(self.settings.lora_rank)

    def suggest_parameters(self, ctx: Context, trial: Trial) -> Parameters:
        layer_count = len(ctx.get_model().get_layers())

        start_layer_index = trial.suggest_int(
            "start_layer_index",
            0,
            layer_count // 2,
        )
        end_layer_index = trial.suggest_int(
            "end_layer_index",
            layer_count // 2,
            layer_count,
        )
        preserve_good_behavior_weight = trial.suggest_float(
            "preserve_good_behavior_weight",
            0.0,
            1.0,
        )
        steer_bad_behavior_weight = trial.suggest_float(
            "steer_bad_behavior_weight",
            0.0001,
            1.0,
            log=True,
        )
        overcorrect_relative_weight = trial.suggest_float(
            "overcorrect_relative_weight",
            0.0,
            1.3,
        )
        neighbor_count = trial.suggest_int(
            "neighbor_count",
            1,
            15,
        )

        return Parameters(
            start_layer_index=start_layer_index,
            end_layer_index=end_layer_index,
            preserve_good_behavior_weight=preserve_good_behavior_weight,
            steer_bad_behavior_weight=steer_bad_behavior_weight,
            overcorrect_relative_weight=overcorrect_relative_weight,
            neighbor_count=neighbor_count,
        )

    def modify_model(self, ctx: Context, parameters: Parameters) -> None:
        model = ctx.get_model()

        for layer_index in range(
            parameters.start_layer_index,
            parameters.end_layer_index,
        ):
            # The model's reset only zeroes B, so A must be re-initialized here,
            # or it would carry over whatever the previous trial left behind.
            # Seeding per layer and on the CPU makes A independent of the layer range,
            # the device, and the global RNG state.
            generator = torch.Generator().manual_seed(
                cast(int, self.heretic_settings.seed) + layer_index
            )

            for component, modules in model.get_layer_modules(layer_index).items():
                for module_index, module in enumerate(modules):
                    # Cast to Linear to access weights and LoRA adapters.
                    module = cast(Linear, module)

                    # We need the base weight in float32 to compute the effective weight.
                    base_weight = cast(Tensor, module.base_layer.weight)
                    quant_state = getattr(base_weight, "quant_state", None)

                    if quant_state is None:
                        W_base = base_weight.to(torch.float32)
                    else:
                        # Use the original dequantization logic from bitsandbytes.
                        W_base = BNB_F.dequantize_4bit(
                            base_weight.data,
                            quant_state,
                        ).to(torch.float32)

                    # Pre-calculate the original row norms to preserve them.
                    # See https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
                    W_row_norms = cast(
                        Tensor,
                        LA.vector_norm(W_base, dim=1, keepdim=True).detach(),
                    )

                    # We optimize the LoRA weights A and B.
                    lora_A = cast(Tensor, module.lora_A["default"].weight)
                    lora_B = cast(Tensor, module.lora_B["default"].weight)

                    # Re-initialize A the same way PEFT does, but deterministically.
                    initial_A = torch.empty(lora_A.shape)
                    torch.nn.init.kaiming_uniform_(
                        initial_A,
                        a=math.sqrt(5),
                        generator=generator,
                    )
                    with torch.no_grad():
                        lora_A.copy_(initial_A)

                    # Move I/O tensors to the device of the adapter weights.
                    good_input, good_output = self.good_module_io[layer_index][
                        component
                    ][module_index]
                    bad_input, bad_output = self.bad_module_io[layer_index][component][
                        module_index
                    ]

                    good_input = good_input.float().to(lora_A.device)
                    good_output = good_output.float().to(lora_A.device)
                    bad_input = bad_input.float().to(lora_A.device)
                    bad_output = bad_output.float().to(lora_A.device)

                    def objective(A: Tensor, B: Tensor) -> Tensor:
                        # Calculate effective weight after applying adapter.
                        W_eff = W_base + (B @ A)

                        if self.settings.preserve_row_magnitudes:
                            # Normalize to unit length, then scale by original norms,
                            # preserving the original row norms.
                            W_eff = F.normalize(W_eff, p=2, dim=1) * W_row_norms

                        # Compute outputs using the effective weight.
                        new_good_output = good_input @ W_eff.T
                        new_bad_output = bad_input @ W_eff.T

                        return ara_loss(
                            good_output,
                            bad_output,
                            new_good_output,
                            new_bad_output,
                            parameters,
                        )

                    optimizer = LBFGS(
                        [lora_A, lora_B],
                        lr=self.settings.learning_rate,
                        max_iter=self.settings.max_iter,
                        history_size=self.settings.history_size,
                        line_search_fn="strong_wolfe",
                    )

                    def closure() -> Tensor:
                        optimizer.zero_grad()
                        loss = objective(lora_A, lora_B)
                        loss.backward()
                        return loss

                    for step in range(self.settings.n_optimization_steps):
                        loss = optimizer.step(closure)
                        if self.settings.print_loss:
                            print(
                                f"\\[{layer_index}/{component}/{module_index}] Step: {step + 1}, Loss: {loss.item():.6f}"
                            )

                    # Free the gradient buffers accumulated during optimization.
                    # Without this, they persist on the model (one full-size gradient
                    # per processed weight) and can easily consume tens of GB of VRAM,
                    # causing out-of-memory errors during the subsequent evaluation.
                    optimizer.zero_grad(set_to_none=True)

    def reset_model(self, ctx: Context) -> None:
        model = ctx.get_model()
        fast_path = model.reset_model()
        if not fast_path:
            model.apply_lora(self.settings.lora_rank)
