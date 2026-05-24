import os
import gradio as gr
from huggingface_hub import InferenceClient

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

client = InferenceClient(
    model=MODEL_ID,
    token=os.environ.get("HF_TOKEN", None),
)


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
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=200,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}\n\nThe model may still be loading. Please try again in 30 seconds."


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
                generate_btn = gr.Button("Generate Docstring ✨", variant="primary")

        with gr.Column(scale=1):
            output = gr.Textbox(
                label="Generated Docstring",
                lines=15,
                max_lines=30,
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
First request may take ~30s to warm up while the model loads.
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