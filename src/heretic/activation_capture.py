import torch
from torch import Tensor
from typing import Dict, List, Any
from collections import defaultdict
from .model import Model
from .utils import Prompt, batchify

def capture_activations(model: Model, prompts: List[Prompt]) -> Dict[str, Tensor]:
    """
    Runs the given prompts through the model and captures the input activations (x)
    for all abliterable components (e.g., o_proj, down_proj) across all layers,
    at the final token position.

    Returns a dict mapping component_name (like '14.attn.o_proj') to a Tensor of shape
    (num_prompts, hidden_dim).
    """
    # A dictionary to store the hooks temporarily
    hook_handles = []
    # A dictionary to accumulate the captured activations
    captured_data = defaultdict(list)

    # We define a hook function generator
    def make_hook(layer_idx: int, component_name: str):
        key = f"{layer_idx}.{component_name}"
        def hook(module, args, kwargs):
            # args[0] is typically the input hidden states: (batch, seq_len, hidden_dim)
            x = args[0]
            if isinstance(x, tuple):
                x = x[0] # Just in case it's a tuple
            # We want the activation at the last token position: x[:, -1, :]
            # We detach it and move it to CPU to save VRAM
            captured_data[key].append(x[:, -1, :].detach().cpu())
        return hook

    # Register the hooks on the model's abliterable components
    for layer_idx in range(len(model.get_layers())):
        for component, modules in model.get_layer_modules(layer_idx).items():
            # For simplicity, we assume one module per component per layer,
            # or if there are multiple (like MoE), we just hook the first one.
            # In the current Heretic design, the same ablation weights apply to all modules
            # for a given component in a layer, so sampling the first expert's input is sufficient.
            # However, for hybrid models, it's safer to capture from all of them and maybe average?
            # Actually, standard dense models only have one module here. Let's just hook all and average.
            for i, module in enumerate(modules):
                # The input is usually the same for all experts in an MoE layer anyway (it's the router's input)
                # But to be safe we use a unique key per module instance if there's >1, or just average.
                # Since the input to all experts in standard MoE is exactly the same, hooking the first is fine.
                handle = module.register_forward_pre_hook(make_hook(layer_idx, component), with_kwargs=True)
                hook_handles.append(handle)
                break # Just hook the first module per component (e.g., standard dense or first expert)

    # Run the prompts through the model to trigger the hooks
    # We use generate to process the prompt, but we only need 1 new token to get the activations
    # at the end of the prompt sequence.
    print(f"* Capturing activations for {len(prompts)} prompts...")
    try:
        # We need to process in batches to avoid OOM
        for batch in batchify(prompts, model.settings.batch_size):
            # We want to do a single forward pass without generating new tokens
            # so the hooks only fire exactly once per prompt.
            chats = [
                [
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ]
                for prompt in batch
            ]
            chat_prompts = model.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=False,
            )
            if model.settings.response_prefix:
                chat_prompts = [p + model.settings.response_prefix for p in chat_prompts]

            inputs = model.tokenizer(
                chat_prompts,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).to(model.model.device)

            with torch.no_grad():
                model.model(**inputs)
    finally:
        # ALWAYS remove the hooks when done!
        for handle in hook_handles:
            handle.remove()

    # Concatenate the accumulated activations
    final_captures = {}
    for key, act_list in captured_data.items():
        # act_list contains tensors of shape (batch_size, hidden_dim)
        # We concatenate them along the batch dimension
        final_captures[key] = torch.cat(act_list, dim=0)

    return final_captures
