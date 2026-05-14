"""
Fine-tune Llama 3.1 8B on Python docstring generation using QLoRA.

Usage (from a Kaggle notebook):
    from src.train import run_training, TrainingConfig
    cfg = TrainingConfig(smoke_test=True)  # 100 examples, 50 steps
    run_training(cfg)
"""
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


@dataclass
class TrainingConfig:
    """All training hyperparameters in one place."""
    
    # Model and dataset
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    dataset_id: str = "Arinkc/pydoc-llama-codesearchnet-curated"
    
    # QLoRA config
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    use_double_quant: bool = True
    
    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    
    # Training hyperparameters
    output_dir: str = "/kaggle/working/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 640
    
    # Logging and evaluation
    logging_steps: int = 25
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    
    # W&B
    wandb_project: str = "pydoc-llama"
    wandb_run_name: Optional[str] = None
    
    # Smoke test mode
    smoke_test: bool = False  # If True, uses 100 examples, 50 steps
    
    def __post_init__(self):
        if self.smoke_test:
            self.num_train_epochs = 1
            self.eval_steps = 25
            self.save_steps = 50
            self.logging_steps = 5
            if self.wandb_run_name is None:
                self.wandb_run_name = "smoke-test"
        else:
            if self.wandb_run_name is None:
                self.wandb_run_name = f"full-run-r{self.lora_r}-lr{self.learning_rate}"

    # HF Hub push
    push_to_hub: bool = True
    hub_repo_id: str = "Arinkc/pydoc-llama-r16-lr2e4"


def load_model_and_tokenizer(cfg: TrainingConfig, hf_token: str):
    """Load Llama 3.1 in 4-bit and configure LoRA."""
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=cfg.use_double_quant,
    )
    
    print(f"Loading tokenizer: {cfg.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model: {cfg.model_id} (4-bit)")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=hf_token,
    )
    
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,
    )
    
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ LoRA configured: {trainable:,} trainable / {total:,} total ({100*trainable/total:.4f}%)")
    
    return model, tokenizer


def load_and_prepare_dataset(cfg: TrainingConfig, tokenizer):
    """Load dataset from HF Hub and pre-tokenize it.
    
    Pre-tokenizing here (rather than letting SFTTrainer do it lazily) avoids
    re-tokenization on every training step.
    """
    print(f"Loading dataset: {cfg.dataset_id}")
    ds = load_dataset(cfg.dataset_id)
    
    if cfg.smoke_test:
        print("⚠️  Smoke test mode — using 100 train / 50 val examples")
        ds['train'] = ds['train'].shuffle(seed=42).select(range(100))
        ds['validation'] = ds['validation'].shuffle(seed=42).select(range(50))
    
    def tokenize_fn(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        result = tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    print("Pre-tokenizing dataset...")
    tokenized = {}
    for split in ['train', 'validation']:
        tokenized[split] = ds[split].map(
            tokenize_fn,
            remove_columns=ds[split].column_names,
            num_proc=4,
            desc=f"Tokenizing {split}",
        )
    
    print(f"   Train: {len(tokenized['train']):,}")
    print(f"   Validation: {len(tokenized['validation']):,}")
    return tokenized


def run_training(cfg: TrainingConfig):
    """Main entry point: load everything, run training, save adapter."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before training")
    
    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    
    model, tokenizer = load_model_and_tokenizer(cfg, hf_token)
    ds = load_and_prepare_dataset(cfg, tokenizer)  # now needs tokenizer
    
    def formatting_func(example):
        """Convert messages list into a tokenizable string using Llama's chat template."""
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    sft_config = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_seq_length=cfg.max_seq_length,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=True,
        fp16=False,
        gradient_checkpointing=False,
        optim="adamw_torch",
        report_to=["wandb"],
        run_name=cfg.wandb_run_name,
    )

    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM, not masked LM
    )
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("🚀 Starting training...")
    trainer.train()
    
    final_dir = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"✅ Final adapter saved locally to {final_dir}")

    # Push to HF Hub if not a smoke test
    if not cfg.smoke_test and cfg.push_to_hub:
        print(f"Pushing to {cfg.hub_repo_id}...")
        trainer.model.push_to_hub(cfg.hub_repo_id, private=False)
        tokenizer.push_to_hub(cfg.hub_repo_id, private=False)
        print(f"✅ Pushed: https://huggingface.co/{cfg.hub_repo_id}")

    return trainer