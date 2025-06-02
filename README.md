
    # Vietnamese Poem to Animation Generator

This project converts Vietnamese poems into animated images using AI models. It performs poem analysis, prompt generation, and image generation in a modular pipeline.

## Features

- Analyze Vietnamese poems using Llama 3.2 model
- Generate image prompts based on poem analysis
- Create animated GIFs using AnimateDiff with emotional themes from the poem

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- Diffusers
- [Optional] Ollama (for local prompt generation)

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install torch transformers diffusers requests
   ```
3. [Optional] Install Ollama for local prompt generation: https://ollama.com/download

## Project Structure

```
poem_utils/                  # Main package
├── __init__.py              # Package initialization
├── analyzer.py              # Poem analysis with Llama 3.2
├── prompt_generator.py      # Image prompt generation
├── diffusion_generator.py   # Animation generation with AnimateDiff
└── pipeline.py              # Pipeline orchestration

poem_to_image_modular.py     # Main application script
```

## Usage

### Command Line

```
python poem_to_image_modular.py [--poem POEM] [--poem-file POEM_FILE] [--output OUTPUT] [--use-local-model]
```

Options:
- `--poem`: Direct poem text input
- `--poem-file`: Path to file containing the poem
- `--output`: Path to save the resulting animation (default: animation.gif)
- `--use-local-model`: Use local model instead of Ollama API for prompt generation

### Example

```
python poem_to_image_modular.py --poem "Trên đầu súng giặc trăng như máu\nRừng cây im như những cột cao\nCao vút kinh hoàng\nRừng che bộ đội\nRừng vây quân thù." --output "poem_animation.gif"
```

### Code Usage

```python
from poem_utils.pipeline import PoemToImagePipeline

# Initialize pipeline
pipeline = PoemToImagePipeline(use_local_model_for_prompt=False)

# Process a poem
poem = """
Trên đầu súng giặc trăng như máu
Rừng cây im như những cột cao
Cao vút kinh hoàng
Rừng che bộ đội
Rừng vây quân thù.
"""
output_path = "animation.gif"
pipeline.process(poem, output_path)
```

## Model Information

This project uses several AI models:

1. **Poem Analysis**: Llama 3.2 model fine-tuned on Vietnamese poetry (kienhoang123/Llama3.2_Poem_Analysis)
2. **Prompt Generation**: Either the same Llama 3.2 model or Ollama API
3. **Animation Generation**: AnimateDiff with Realistic Vision V5.1

## T5 Model Details

The project also includes a T5 model fine-tuned on kienhoang123/Vietnamese_Poem_Analysis_VN to analyze Vietnamese poetry.

- **Base Model**: t5-small
- **Training Data**: Vietnamese poem analysis dataset
- **Tasks**: Extract emotion, metaphor, setting, motion, and prompt from Vietnamese poems

### T5 Model Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("kienhoang123/ViT5_Poem_Analysis")
model = AutoModelForSeq2SeqLM.from_pretrained("kienhoang123/ViT5_Poem_Analysis")

poem = "Your Vietnamese poem here"
inputs = tokenizer(poem, return_tensors="pt")
outputs = model.generate(**inputs)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

The output is formatted as: "emotion ||| metaphor ||| setting ||| motion ||| prompt"

## License

Apache License 2.0
    