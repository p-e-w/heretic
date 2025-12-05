# Heretic Plugin System Guide

Heretic v1.0+ introduces a flexible **Plugin System** that allows you to define custom optimization objectives for "abliteration" (OR steering/tweaking). Instead of just removing refusals, you can now steer the model to minimize or maximize *any* measurable property of its output.

## Core Concept

The core of Heretic is an optimization loop that finds a direction in the model's latent space to ablate (remove). The "Plugin" defines **what to remove**.

-   **Refusal Plugin (Default/Built-in)**: Scores responses based on the presence of refusal phrases ("I cannot", "Sorry"). Heretic minimizes this score.
-   **Classifier Plugin**: Scores responses based on a target label from a Hugging Face classification model (e.g., "Joy"). Heretic can NOW minimize or maximize this.
-   **Custom Plugins**: You can write your own Python class to score responses however you like (e.g., length, regex, external API, another LLM).

## Using Built-in Plugins

You select a plugin using the `--plugin` argument and pass configuration using `--plugin-args`.

### 1. Refusal Plugin (Default)
Removes refusal mechanisms.

```bash
heretic --model "meta-llama/Llama-3.2-3B-Instruct"
```

### 2. Classifier Plugin
Steers the model's tone or personality using a text classification model.

**Example: Make the model Happier (Maximize "Joy")**
```bash
heretic --model "meta-llama/Llama-3.2-3B-Instruct" \
    --plugin classifier \
    --plugin-args '{"model_name": "j-hartmann/emotion-english-distilroberta-base", "target_label": "joy", "minimize": false}'
```

**Example: Remove Sadness (Minimize "Sadness")**
```bash
heretic --model "meta-llama/Llama-3.2-3B-Instruct" \
    --plugin classifier \
    --plugin-args '{"model_name": "j-hartmann/emotion-english-distilroberta-base", "target_label": "sadness", "minimize": true}'
```

## Creating Custom Plugins

You can create a custom plugin by writing a Python file (e.g., `my_plugin.py`).

### Requirements
1.  Import `Plugin` from `heretic.plugin_interface`.
2.  Create a class that inherits from `Plugin`.
3.  Implement the `score(self, responses: list[str]) -> list[float]` method.
4.  (Optional) Implement the `minimize` property (defaults to `True`).
5.  **Crucial**: Assign your class to a global variable named `PLUGIN_CLASS`.

### Example: `BrevityPlugin` (Penalize long responses)

Save this as `brevity_plugin.py`:

```python
from heretic.plugin_interface import Plugin

class BrevityPlugin(Plugin):
    def __init__(self, max_length: int = 50):
        self.max_length = int(max_length)

    @property
    def minimize(self) -> bool:
        # We want to minimize the "badness" (excessive length)
        return True

    def score(self, responses: list[str]) -> list[float]:
        scores = []
        for response in responses:
            # Simple score: 1.0 if too long, 0.0 if okay
            # Or continuous: normalized length
            length = len(response.split())
            if length > self.max_length:
                scores.append(1.0)
            else:
                scores.append(0.0)
        return scores

# Export the class
PLUGIN_CLASS = BrevityPlugin
```

### Running a Custom Plugin

Pass the path to your python file:

```bash
heretic --model "meta-llama/Llama-3.2-3B-Instruct" \
    --plugin "./brevity_plugin.py" \
    --plugin-args '{"max_length": "20"}'
```

## Arguments Format
The `--plugin-args` argument accepts a JSON string or a dictionary-like string format supported by the configuration loader. Ensure keys match the `__init__` arguments of your plugin class.
