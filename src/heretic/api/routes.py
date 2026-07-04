# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
from importlib.metadata import version

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)

from ..system import get_accelerator_info_dict
from .auth import get_hf_token, websocket_authorized
from .conversion import ablate_request_to_settings
from .models import (
    AblateRequest,
    AblateResponse,
    AblationResult,
    AcceleratorInfo,
    ChatRequest,
    ChatResponse,
    ExportRequest,
    ExportResponse,
    ExportStatusResponse,
    HealthResponse,
    MessageResponse,
    SelectTrialRequest,
    SelectTrialResponse,
    TaskStatus,
    TaskStatusResponse,
)
from .tasks import task_manager

router = APIRouter()


def _require_task(task_id: str):
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    accel = get_accelerator_info_dict()
    active = task_manager.active_task
    return HealthResponse(
        status="ok",
        version=version("heretic-llm"),
        accelerator=AcceleratorInfo(
            type=accel.get("type"),
            devices=[
                {"name": d.get("name"), "vram_gb": d.get("vram_gb")}
                for d in accel.get("devices", [])
            ],
        ),
        model_loaded=task_manager.has_active_model,
        active_task=active.task_id if active else None,
        huggingface_upload_enabled=get_hf_token() is not None,
    )


@router.post("/ablate", response_model=AblateResponse, status_code=202)
async def start_ablation(
    request: AblateRequest, background_tasks: BackgroundTasks
) -> AblateResponse:
    active = task_manager.active_task
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"An abliteration task ({active.task_id}) is already running or "
                "pending. Cancel it before starting a new one."
            ),
        )

    settings = ablate_request_to_settings(request)
    task = task_manager.create_task(settings)
    background_tasks.add_task(task.run)

    return AblateResponse(
        task_id=task.task_id,
        status=TaskStatus.PENDING,
        message="Abliteration task created and queued.",
    )


@router.get("/ablate/{task_id}", response_model=TaskStatusResponse)
async def get_ablation_status(task_id: str) -> TaskStatusResponse:
    task = _require_task(task_id)
    return TaskStatusResponse(**task.status_response)


@router.get("/ablate/{task_id}/results", response_model=AblationResult)
async def get_ablation_results(task_id: str) -> AblationResult:
    task = _require_task(task_id)
    return AblationResult(**task.result)


@router.post("/ablate/{task_id}/select-trial", response_model=SelectTrialResponse)
async def select_trial(
    task_id: str, request: SelectTrialRequest
) -> SelectTrialResponse:
    task = _require_task(task_id)
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Task is not completed yet")

    loop = asyncio.get_running_loop()
    info = await loop.run_in_executor(None, task.select_trial, request.trial_index)
    if info is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No Pareto-optimal trial with index {request.trial_index}. "
                f"Valid indices: {task.pareto_indices()}."
            ),
        )

    return SelectTrialResponse(
        selected_trial=info, message=f"Trial {info.index} applied to model."
    )


@router.post("/ablate/{task_id}/cancel", response_model=MessageResponse)
async def cancel_ablation(task_id: str) -> MessageResponse:
    task = _require_task(task_id)
    if task.status not in (TaskStatus.RUNNING, TaskStatus.PENDING):
        raise HTTPException(status_code=409, detail="Task is not running or pending")

    task.cancel()
    return MessageResponse(message="Cancellation requested.")


@router.delete("/ablate/{task_id}", response_model=MessageResponse)
async def delete_ablation(task_id: str) -> MessageResponse:
    task = _require_task(task_id)
    if task.status in (TaskStatus.RUNNING, TaskStatus.PENDING):
        raise HTTPException(
            status_code=409,
            detail="Task is still active; cancel it before deleting.",
        )
    if task.export_in_progress:
        raise HTTPException(
            status_code=409,
            detail="An export for this task is in progress; wait for it to finish.",
        )

    task_manager.remove_task(task_id)
    return MessageResponse(message="Task deleted and resources released.")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    task = task_manager.get_task_with_model()
    if task is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "No abliterated model available. Run an abliteration task and "
                "select a trial first."
            ),
        )

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None, task.chat, request.message, request.system_prompt, request.max_tokens
    )
    return ChatResponse(response=response)


@router.post("/ablate/{task_id}/export", response_model=ExportResponse, status_code=202)
async def export_model(
    task_id: str, request: ExportRequest, background_tasks: BackgroundTasks
) -> ExportResponse:
    task = _require_task(task_id)
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Task is not completed yet")
    if not task.has_model:
        raise HTTPException(
            status_code=409,
            detail="No model available for this task. Its resources may have been released.",
        )
    if not task.has_selected_trial:
        raise HTTPException(
            status_code=409,
            detail="No trial selected. Call select-trial before exporting.",
        )
    if task_manager.active_task is not None:
        raise HTTPException(
            status_code=409,
            detail="An abliteration task is active; exporting would contend for the GPU.",
        )
    if task_manager.export_in_progress:
        raise HTTPException(
            status_code=409,
            detail="An export is already in progress. Wait for it to finish.",
        )

    destination = request.destination.lower()
    if destination not in ("local", "huggingface"):
        raise HTTPException(
            status_code=400,
            detail='Invalid destination. Use "local" or "huggingface".',
        )

    export_kwargs: dict[str, object] = {"max_shard_size": request.max_shard_size}

    if destination == "local":
        if not request.save_directory:
            raise HTTPException(
                status_code=400,
                detail="'save_directory' is required for local export.",
            )
        export_kwargs["save_directory"] = request.save_directory
        message = "Local export started."
    else:
        token = get_hf_token()
        if token is None:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Hugging Face uploads are disabled. Start the server with "
                    "--hf-token or set the HF_TOKEN environment variable to enable them."
                ),
            )
        if not request.repo_id:
            raise HTTPException(
                status_code=400,
                detail="'repo_id' is required for Hugging Face export.",
            )
        export_kwargs["repo_id"] = request.repo_id
        export_kwargs["token"] = token
        export_kwargs["private"] = request.private
        message = "Hugging Face export started."

    job = task.start_export(destination, request.strategy.value)
    background_tasks.add_task(task.run_export, job, **export_kwargs)

    return ExportResponse(
        export_id=job.export_id,
        task_id=task.task_id,
        status=TaskStatus.PENDING,
        message=message,
    )


@router.get("/exports/{export_id}", response_model=ExportStatusResponse)
async def get_export_status(export_id: str) -> ExportStatusResponse:
    result = task_manager.get_export_job(export_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Export job not found")
    _task, job = result
    return ExportStatusResponse(**job.status_response)


@router.websocket("/ablate/{task_id}/ws")
async def ablation_websocket(websocket: WebSocket, task_id: str) -> None:
    if not await websocket_authorized(websocket):
        # Reject before completing the handshake so the client sees a failure.
        await websocket.close(code=1008)  # Policy violation.
        return

    await websocket.accept()

    task = task_manager.get_task(task_id)
    if task is None:
        await websocket.send_json(
            {"type": "error", "data": {"detail": "Task not found"}}
        )
        await websocket.close()
        return

    terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    last_progress: dict[str, object] = {}

    try:
        while True:
            current = task.progress  # already a thread-safe snapshot copy
            if current != last_progress:
                await websocket.send_json({"type": "progress", "data": current})
                last_progress = current

            if task.status in terminal:
                if task.status == TaskStatus.COMPLETED:
                    await websocket.send_json(
                        {"type": "completed", "data": task.result}
                    )
                elif task.status == TaskStatus.FAILED:
                    await websocket.send_json(
                        {"type": "failed", "data": {"error": task.error}}
                    )
                else:
                    await websocket.send_json({"type": "cancelled", "data": {}})
                break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()
