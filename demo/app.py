import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_ID = "Arinkc/pydoc-llama-r16-full"

HF_TOKEN = os.environ.get("HF_TOKEN")

SYSTEM_PROMPT = (
    "You are an expert Python documentation writer. "
    "Given a Python function, generate a concise, "
    "Google-style docstring. Output only the docstring "
    "text with no markdown, no code fences, and no explanation."
)

DESCRIPTION = """
# PyDocLlama — AI Python Documentation Generator

Generate Google-style Python docstrings using a fine-tuned Llama 3.1 8B model.

**Model:** Arinkc/pydoc-llama-r16-full  
Fine-tuned with QLoRA on 22,473 curated Python function/docstring pairs.

### Key Improvements Over Base Model
- Hallucinated exceptions: 11% → 0%
- Verbose outputs: 19.5% → 0%
- Format compliance: 80.5% → 100%

⚠️ Running on free CPU hardware.  
First startup may take several minutes.  
Generation may take 30–90 seconds.
"""

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    token=HF_TOKEN,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model... this may take several minutes.")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
    token=HF_TOKEN,
)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_ID,
    token=HF_TOKEN,
)

model.eval()

print("✅ Model loaded successfully")


def generate_docstring(function_code: str) -> str:
    """Generate a Google-style Python docstring."""

    if not function_code.strip():
        return "Please enter a Python function."

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Generate a Google-style docstring for this function:\n\n"
                f"```python\n{function_code}\n```"
            ),
        },
    ]

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        return response

    except Exception as e:
        return f"Generation failed:\n\n{str(e)}"


with gr.Blocks(title="PyDocLlama") as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():

        with gr.Column(scale=2):

            code_input = gr.Textbox(
                label="Python Function",
                lines=18,
                max_lines=30,
            )

            with gr.Row():

                clear_btn = gr.Button(
                    "Clear",
                    variant="secondary",
                )

                generate_btn = gr.Button(
                    "Generate Docstring ✨",
                    variant="primary",
                )

        with gr.Column(scale=1):

            output = gr.Textbox(
                label="Generated Docstring",
                lines=18,
                max_lines=30,
            )

    gr.Markdown(
        """
## Example Function

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
"""
)
    gr.Markdown(
        """
About This Project

PyDocLlama was fine-tuned using QLoRA on an NVIDIA A100 GPU.

Training:

22,473 curated Python examples
LoRA rank: 16
4-bit NF4 quantization
Final training loss: 0.63

This demo runs entirely on free CPU hardware for public access.
"""
)

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