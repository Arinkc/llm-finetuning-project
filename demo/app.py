import os
import torch
import gradio as gr
from transformers import pipeline, AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "Arinkc/pydoc-llama-r16-merged"

SYSTEM_PROMPT = (
    "You are an expert Python documentation writer. Given a Python function, "
    "generate a concise, Google-style docstring. Output only the docstring "
    "text—no surrounding code, no markdown formatting, no preamble."
)

DESCRIPTION = """
# PyDocLlama — AI Python Documentation Generator

Generate Google-style Python docstrings using a fine-tuned Llama 3.1 8B model.

**Model:** [Arinkc/pydoc-llama-r16-full](https://huggingface.co/Arinkc/pydoc-llama-r16-full)
Fine-tuned with QLoRA on 22,473 curated Python function/docstring pairs.

**Key improvements over base model (200 held-out test examples):**
- Hallucinated exceptions eliminated: 11% → 0%
- Verbose outputs eliminated: 19.5% → 0%
- Format compliance: 80.5% → 100%

[GitHub](https://github.com/arinkc/llm-finetuning-project) |
[Dataset](https://huggingface.co/datasets/Arinkc/pydoc-llama-codesearchnet-curated) |
[Model](https://huggingface.co/Arinkc/pydoc-llama-r16-full)
"""

EXAMPLES = [
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "def merge_dicts(dict1, dict2, overwrite=True):\n    result = dict1.copy()\n    for key, value in dict2.items():\n        if key not in result or overwrite:\n            result[key] = value\n    return result",
]

HF_TOKEN = os.environ.get("HF_TOKEN", None)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
)
model.eval()
print("Model loaded.")


def generate_docstring(function_code: str) -> str:
    if not function_code.strip():
        return "Please enter a Python function."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Generate a Google-style docstring for this function:\n\n```python\n{function_code}\n```",
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            temperature=1.0,
        )

    response = tokenizer.decode(
        outputs[0][inputs.shape[1]:],
        skip_special_tokens=True,
    ).strip()

    return response


with gr.Blocks(title="PyDocLlama") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=2):
            code_input = gr.Textbox(
                label="Python Function",
                lines=15,
                max_lines=30,
            )
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                generate_btn = gr.Button(
                    "Generate Docstring ✨",
                    variant="primary",
                )

        with gr.Column(scale=1):
            output = gr.Textbox(
                label="Generated Docstring",
                lines=15,
                max_lines=30,
                show_copy_button=True,
            )

    with gr.Accordion("Examples — click to load", open=False):
        for example in EXAMPLES:
            gr.Button(example[:60] + "...").click(
                fn=lambda e=example: e,
                outputs=code_input,
            )

    gr.Markdown("""
---
**About:** Fine-tuned Llama 3.1 8B with QLoRA on an A100 GPU.
Training loss: 2.3 → 0.63 over 4,212 steps.
Generation may take 30-60 seconds on CPU.
    """)

    generate_btn.click(
        fn=generate_docstring,
        inputs=code_input,
        outputs=output,
    )
    clear_btn.click(
        fn=lambda: ("", ""),
        outputs=[code_input, output],
    )

demo.launch()