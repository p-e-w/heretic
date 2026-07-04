# SPDX-License-Identifier: AGPL-3.0-or-later

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetSpec(BaseModel):
    """A dataset source for prompts (mirrors ``config.DatasetSpecification``)."""

    dataset: str = Field(..., description="Hugging Face dataset ID or local path.")
    split: str | None = Field(
        default=None, description="Portion of the dataset to use."
    )
    column: str | None = Field(
        default=None, description="Column containing the prompts."
    )
    prefix: str = Field(default="", description="Text to prepend to each prompt.")
    suffix: str = Field(default="", description="Text to append to each prompt.")
    system_prompt: str | None = Field(
        default=None, description="System prompt override."
    )


class TrialInfo(BaseModel):
    index: int
    refusals: int
    kl_divergence: float
    direction_index: float | None
    parameters: dict[str, dict[str, float]]


class AblateRequest(BaseModel):
    """Parameters for an abliteration run. All fields have sensible defaults."""

    model: str = Field(..., description="Hugging Face model ID or local path.")
    model_commit: str | None = None

    # Loading / hardware
    quantization: str = Field(default="none", description='"none" or "bnb_4bit".')
    device_map: str | dict[str, Any] = "auto"
    max_memory: dict[str, str] | None = None
    offload_outputs_to_cpu: bool = True
    batch_size: int = Field(default=0, ge=0, description="0 = auto-detect.")
    max_batch_size: int = Field(default=128, gt=0)

    # Optimization
    n_trials: int = Field(default=200, gt=0)
    n_startup_trials: int = Field(default=60, ge=0)
    seed: int | None = None
    study_checkpoint_dir: str = "checkpoints"

    # Abliteration behaviour
    kl_divergence_scale: float = 1.0
    kl_divergence_target: float = 0.01
    orthogonalize_direction: bool = True
    row_normalization: str = Field(
        default="full", description='"none", "pre", or "full".'
    )
    winsorization_quantile: float = 1.0
    max_response_length: int = Field(default=100, gt=0)

    # Prompting
    system_prompt: str = "You are a helpful assistant."
    good_prompts: DatasetSpec = DatasetSpec(
        dataset="mlabonne/harmless_alpaca", split="train[:400]", column="text"
    )
    bad_prompts: DatasetSpec = DatasetSpec(
        dataset="mlabonne/harmful_behaviors", split="train[:400]", column="text"
    )
    good_evaluation_prompts: DatasetSpec = DatasetSpec(
        dataset="mlabonne/harmless_alpaca", split="test[:100]", column="text"
    )
    bad_evaluation_prompts: DatasetSpec = DatasetSpec(
        dataset="mlabonne/harmful_behaviors", split="test[:100]", column="text"
    )


class AblateResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: dict[str, Any] = {}
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class AblationResult(BaseModel):
    task_id: str
    status: TaskStatus
    pareto_trials: list[TrialInfo] = []
    selected_trial: TrialInfo | None = None
    base_refusals: int | None = None
    n_bad_prompts: int | None = None
    error: str | None = None


class SelectTrialRequest(BaseModel):
    trial_index: int = Field(
        ...,
        ge=0,
        description=(
            "The 'index' of the trial to apply, as reported in each entry of "
            "the /results 'pareto_trials' list."
        ),
    )


class SelectTrialResponse(BaseModel):
    selected_trial: TrialInfo
    message: str


class ChatRequest(BaseModel):
    message: str
    system_prompt: str | None = None
    max_tokens: int = Field(default=4096, gt=0)


class ChatResponse(BaseModel):
    response: str


class ExportStrategy(str, Enum):
    MERGE = "merge"
    ADAPTER = "adapter"


class ExportRequest(BaseModel):
    """Request to export the currently selected abliterated model.

    Local export (``destination="local"``) is always available. Uploading to
    the Hugging Face Hub (``destination="huggingface"``) is only permitted when
    the server was started with a Hugging Face token.
    """

    destination: str = Field(
        default="local",
        description='Where to export: "local" or "huggingface".',
    )
    strategy: ExportStrategy = Field(
        default=ExportStrategy.MERGE,
        description=(
            'How to export: "merge" (LoRA merged into a full model) or '
            '"adapter" (LoRA adapter only).'
        ),
    )
    # Local export.
    save_directory: str | None = Field(
        default=None,
        description="Target directory for local export (required when destination=local).",
    )
    # Hugging Face export.
    repo_id: str | None = Field(
        default=None,
        description="Target repository ID (required when destination=huggingface).",
    )
    private: bool = Field(
        default=True,
        description="Whether the created Hugging Face repository should be private.",
    )
    max_shard_size: str = Field(
        default="5GB",
        description="Maximum size for individual safetensors shards.",
    )


class ExportResponse(BaseModel):
    """Returned when an export job is accepted; the work runs in the background."""

    export_id: str
    task_id: str
    status: TaskStatus
    message: str


class ExportStatusResponse(BaseModel):
    export_id: str
    task_id: str
    status: TaskStatus
    destination: str
    strategy: ExportStrategy
    location: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class MessageResponse(BaseModel):
    message: str


class AcceleratorInfo(BaseModel):
    type: str | None = None
    devices: list[dict[str, Any]] = []


class HealthResponse(BaseModel):
    status: str
    version: str
    accelerator: AcceleratorInfo = AcceleratorInfo()
    model_loaded: bool = False
    active_task: str | None = None
    huggingface_upload_enabled: bool = False
