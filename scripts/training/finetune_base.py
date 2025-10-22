"""
StoryForge Base Model Fine-tuning with LoRA
Fine-tunes Llama 3.1 8B on creative writing data
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

import wandb


@dataclass
class ModelConfig:
    """Configuration for model and training"""

    # Model settings
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    use_4bit: bool = True  # 4-bit quantization for memory efficiency

    # LoRA settings
    # lora_r: int = 16  # Rank of LoRA matrices
    # lora_alpha: int = 32  # Scaling parameter
    lora_r: int = 8  # Was 16
    lora_alpha: int = 16  # Was 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: [
            "q_proj",  # Query projection in attention
            "k_proj",  # Key projection
            "v_proj",  # Value projection
            "o_proj",  # Output projection
            "gate_proj",  # Gate projection in FFN
            "up_proj",  # Up projection in FFN
            "down_proj",  # Down projection in FFN
        ]
    )

    # Data settings
    data_path: str = "data/processed"
    max_seq_length: int = 2048
    # max_seq_length: int = 1024  # Was 2048

    # Training settings
    output_dir: str = "models/checkpoints/base_model"
    num_train_epochs: int = 3
    # per_device_train_batch_size: int = 2  # Small for memory
    # gradient_accumulation_steps: int = 8  # Effective batch size = 2*8 = 16
    per_device_train_batch_size: int = 1  # Was 2
    gradient_accumulation_steps: int = 16  # Was 8
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # Optimization
    optim: str = "paged_adamw_8bit"  # Memory-efficient optimizer
    gradient_checkpointing: bool = True
    # TODO: Set to true on Colab
    fp16: bool = False  # Set to True if you have CUDA
    bf16: bool = False

    # Misc
    seed: int = 42
    use_wandb: bool = True
    wandb_project: str = "storyforge"
    wandb_run_name: str = "base_lora_finetune"


def get_device_config():
    """Get appropriate device configuration for current platform"""
    import torch

    if torch.cuda.is_available():
        # CUDA (Linux/Windows with NVIDIA GPU)
        return {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
    elif torch.backends.mps.is_available():
        # Apple Silicon
        return {
            "device_map": {"": "mps"},
            "torch_dtype": torch.float16,
        }
    else:
        # CPU fallback
        return {
            "device_map": {"": "cpu"},
            "torch_dtype": torch.float32,  # CPU works better with float32
        }


class StoryForgeTrainer:
    """Handles fine-tuning of Llama with LoRA"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"🔧 Using device: {self.device}")

    def load_model_and_tokenizer(self):
        """Load base model and tokenizer with quantization"""
        print(f"\n📥 Loading model: {self.config.model_name}")

        # Quantization config for memory efficiency
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Then use it:
        device_config = get_device_config()

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            **device_config,
        )

        # Prepare model for k-bit training
        if self.config.use_4bit:
            model = prepare_model_for_kbit_training(model)

        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Required for training

        print("✅ Model loaded")
        print(f"   Parameters: {model.num_parameters() / 1e9:.2f}B")
        print(f"   Memory: {model.get_memory_footprint() / 1e9:.2f} GB")

        return model, tokenizer

    def setup_lora(self, model):
        """Configure and apply LoRA adapters"""
        print("\n🎯 Setting up LoRA adapters...")
        print(f"   Rank (r): {self.config.lora_r}")
        print(f"   Alpha: {self.config.lora_alpha}")
        print(f"   Target modules: {', '.join(self.config.lora_target_modules)}")

        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)

        # Print trainable parameters
        trainable_params, all_param = model.get_nb_trainable_parameters()
        print("✅ LoRA applied")
        print(
            f"   Trainable params: {trainable_params:,} ({100 * trainable_params / all_param:.2f}%)"
        )
        print(f"   Total params: {all_param:,}")
        print(f"   Memory reduction: ~{all_param / trainable_params:.0f}x")

        return model

    def load_datasets(self, tokenizer):
        """Load and tokenize training data"""
        print(f"\n📚 Loading datasets from {self.config.data_path}")

        # Load datasets
        data_files = {
            "train": str(Path(self.config.data_path) / "train.jsonl"),
            "validation": str(Path(self.config.data_path) / "val.jsonl"),
        }

        dataset = load_dataset("json", data_files=data_files)

        print(f"   Train samples: {len(dataset['train']):,}")
        print(f"   Validation samples: {len(dataset['validation']):,}")

        # Tokenize function
        def tokenize_function(examples):
            # Tokenize texts
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,  # Dynamic padding in collator
                return_tensors=None,
            )
            return tokenized

        # Tokenize datasets
        print("   Tokenizing...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing",
        )

        print("✅ Datasets ready")

        return tokenized_dataset

    def train(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("🚀 StoryForge Fine-tuning with LoRA")
        print("=" * 60 + "\n")

        # Initialize wandb if enabled
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__,
            )

        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()

        # Setup LoRA
        model = self.setup_lora(model)

        # Load datasets
        tokenized_dataset = self.load_datasets(tokenizer)

        # Data collator for dynamic padding
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # We're doing CLM, not MLM
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim=self.config.optim,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            report_to="wandb" if self.config.use_wandb else "none",
            seed=self.config.seed,
            dataloader_num_workers=0,  # Set to 0 for Mac
            remove_unused_columns=False,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
        )

        # Train!
        print("\n🔥 Starting training...")
        print(f"   Epochs: {self.config.num_train_epochs}")
        print(f"   Batch size: {self.config.per_device_train_batch_size}")
        print(f"   Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(
            f"   Effective batch size: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}"
        )
        print(f"   Learning rate: {self.config.learning_rate}")
        print()

        trainer.train()

        # Save final model
        print("\n💾 Saving final model...")
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)

        # Save LoRA config
        lora_config_path = Path(self.config.output_dir) / "lora_config.json"
        with open(lora_config_path, "w") as f:
            json.dump(
                {
                    "lora_r": self.config.lora_r,
                    "lora_alpha": self.config.lora_alpha,
                    "lora_dropout": self.config.lora_dropout,
                    "target_modules": self.config.lora_target_modules,
                },
                f,
                indent=2,
            )

        print(f"✅ Model saved to {self.config.output_dir}")
        print("\n" + "=" * 60)
        print("🎉 Training Complete!")
        print("=" * 60)

        if self.config.use_wandb:
            wandb.finish()


def main():
    """Main execution"""

    # Add this to your config
    if os.getenv("COLAB_GPU"):  # Colab environment
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    else:  # Local testing
        model_name = "meta-llama/Llama-3.2-1B-Instruct"

    print(
        """
    ╔════════════════════════════════════════════╗
    ║       StoryForge LoRA Fine-tuning         ║
    ║                                            ║
    ║  Fine-tuning Llama 3.1 8B on creative     ║
    ║  writing using LoRA adapters              ║
    ╚════════════════════════════════════════════╝
    """
    )

    # Create config
    if torch.backends.mps.is_available():
        config = ModelConfig(use_4bit=False, model_name=model_name)
    else:
        config = ModelConfig(model_name=model_name)

    # Create trainer
    trainer = StoryForgeTrainer(config)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
