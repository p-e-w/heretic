
import sys
import torch
import torch.nn as nn
from typing import Any, Type, cast, Optional

# Mock the parts of heretic we need
class MockModuleList(nn.ModuleList):
    pass

class MockLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

class MockLayer(nn.Module):
    def __init__(self, is_mamba=False):
        super().__init__()
        if is_mamba:
            self.mixer = nn.Module()
            self.mixer.out_proj = MockLinear(512, 512)
        else:
            self.self_attn = nn.Module()
            self.self_attn.o_proj = MockLinear(512, 512)
            self.mlp = nn.Module()
            self.mlp.down_proj = MockLinear(2048, 512)

class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = MockModuleList([
            MockLayer(is_mamba=False),
            MockLayer(is_mamba=True),
            MockLayer(is_mamba=False)
        ])

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MockBackbone()
        self.config = type('Config', (), {'hidden_size': 512, 'name_or_path': 'mock-hybrid'})()
        self.device = torch.device("cpu")
        self.dtype = torch.float32

# The logic from model.py for testing
def get_layers_logic(model):
    # Check for common hybrid model structures (e.g., Falcon-H1R, Mamba-2)
    # Search for any attribute that is a ModuleList and likely contains the layers.
    # (Simulating the logic I wrote in model.py)
    
    # Check common locations
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    
    # Generic search
    for name, module in model.named_modules():
        if isinstance(module, (nn.ModuleList, MockModuleList)) and name.endswith(".layers"):
            return module

    if hasattr(model, "layers") and isinstance(model.layers, (nn.ModuleList, MockModuleList)):
        return model.layers
    
    return None

def get_layer_modules_logic(layer, hidden_size):
    modules = {}
    seen_ids = set()

    def try_add(component, module):
        if not isinstance(module, nn.Module): return
        mod_id = id(module)
        if mod_id in seen_ids: return
        seen_ids.add(mod_id)
        
        if component == "ssm.out_proj" and hidden_size is not None:
            out_features = getattr(module, "out_features", None)
            if out_features is not None and out_features != hidden_size:
                return

        modules.setdefault(component, []).append(module)

    # Attention
    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
        try_add("attn.o_proj", layer.self_attn.o_proj)

    # MLP
    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
        try_add("mlp.down_proj", layer.mlp.down_proj)

    # SSM (The new robust logic)
    for name, module in layer.named_modules():
        name_lower = name.lower()
        if any(x in name_lower for x in ["ssm", "mamba", "mixer", "recurrent"]):
            for subname, submod in module.named_modules():
                if subname.endswith(("out_proj", "output", "proj")) and isinstance(submod, nn.Module):
                    try_add("ssm.out_proj", submod)

    return modules

# RUN TESTS
model = MockModel()
layers = get_layers_logic(model)

if layers is None:
    print("FAILED: Could not find layers")
    sys.exit(1)

print(f"Found {len(layers)} layers")

for i, layer in enumerate(layers):
    modules = get_layer_modules_logic(layer, 512)
    print(f"Layer {i}: {list(modules.keys())}")
    for k, v in modules.items():
        print(f"  {k}: {len(v)} module(s)")

# Verification: Layer 1 should have ssm.out_proj
if "ssm.out_proj" in get_layer_modules_logic(layers[1], 512):
    print("\nSUCCESS: Mamba layer detected correctly!")
else:
    print("\nFAILED: Mamba layer NOT detected.")
    sys.exit(1)

# Verification: Layer 0 should have attn.o_proj and mlp.down_proj
l0_mods = get_layer_modules_logic(layers[0], 512)
if "attn.o_proj" in l0_mods and "mlp.down_proj" in l0_mods:
    print("SUCCESS: Transformer layer detected correctly!")
else:
    print("FAILED: Transformer layer parts missing.")
    sys.exit(1)
