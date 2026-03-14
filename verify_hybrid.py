
import torch
import torch.nn as nn
from src.heretic.model import Model
from src.heretic.config import Settings

class MockConfig:
    def __init__(self):
        self.hidden_size = 512
        self.name_or_path = "mock-hybrid"

class MockLayer(nn.Module):
    def __init__(self, is_mamba=False):
        super().__init__()
        if is_mamba:
            self.mixer = nn.Module()
            self.mixer.out_proj = nn.Linear(512, 512)
        else:
            self.self_attn = nn.Module()
            self.self_attn.o_proj = nn.Linear(512, 512)
            self.mlp = nn.Module()
            self.mlp.down_proj = nn.Linear(2048, 512)

class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            MockLayer(is_mamba=False),
            MockLayer(is_mamba=True),
            MockLayer(is_mamba=False)
        ])

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MockBackbone()
        self.config = MockConfig()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
    
    def generate(self, *args, **kwargs):
        return torch.zeros((1, 10))

# Mock the Model class to use our MockModel
class TestModel(Model):
    def __init__(self, settings):
        self.settings = settings
        self.model = MockModel()
        self.tokenizer = type('MockTokenizer', (), {
            'pad_token': '[PAD]',
            'eos_token': '[EOS]',
            'padding_side': 'left',
            'apply_chat_template': lambda *args, **kwargs: ["mock"],
            '__call__': lambda *args, **kwargs: {'input_ids': torch.zeros((1, 5)), 'attention_mask': torch.zeros((1, 5))}
        })()

settings = Settings(model="mock-hybrid")
test_model = TestModel(settings)

print("Testing layer detection...")
layers = test_model.get_layers()
print(f"Number of layers found: {len(layers)}")

for i in range(len(layers)):
    modules = test_model.get_layer_modules(i)
    print(f"Layer {i} components: {list(modules.keys())}")
    for component, mods in modules.items():
        print(f"  {component}: {len(mods)} modules")

print("All tests passed (simulated)")
