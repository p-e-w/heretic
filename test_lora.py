
from heretic.config import QuantizationMethod
from heretic.model import Model


class TestSettings:
    model = 'HuggingFaceM4/tiny-random-LlamaForCausalLM'
    evaluate_model = None
    dtypes = ['auto']
    device_map = 'auto'
    trust_remote_code = None
    quantization = QuantizationMethod.BNB_4BIT
    batch_size = 1
    max_batch_size = 1
    max_response_length = 10
    system_prompt = 'Test'

settings = TestSettings()
m = Model(settings)

print("\n=== Testing get_layer_modules() after fix ===")
modules = m.get_layer_modules(0)
for comp, mods in modules.items():
    print(f'{comp}:')
    for mod in mods:
        print(f'  type: {type(mod).__name__}')
        print(f'  has lora_A: {hasattr(mod, "lora_A")}')
