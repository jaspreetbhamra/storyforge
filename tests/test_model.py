"""
Test your fine-tuned StoryForge model
Compare base vs fine-tuned generation
"""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_path, use_lora=True):
    """Load model with or without LoRA adapters"""
    print(f"\n📥 Loading model from {model_path}...")

    base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    if use_lora:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, model_path)
        print("✅ Loaded model with LoRA adapters")
    else:
        # Load base model only (for comparison)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("✅ Loaded base model (no fine-tuning)")

    model.eval()
    return model, tokenizer


def generate_story(model, tokenizer, prompt, max_new_tokens=500, temperature=0.8):
    """Generate a story from a prompt"""

    # Format prompt in Llama 3.1 instruction format
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a creative writer. Write engaging, well-structured stories based on the given prompts.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the story (remove prompt)
    story = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[
        -1
    ].strip()

    return story


def compare_models(base_model_path, finetuned_model_path, test_prompts):
    """Compare base vs fine-tuned model on test prompts"""

    print("\n" + "=" * 80)
    print("🎭 StoryForge Model Comparison: Base vs Fine-tuned")
    print("=" * 80)

    # Load models
    print("\n📦 Loading base model...")
    base_model, base_tokenizer = load_model(base_model_path, use_lora=False)

    print("\n📦 Loading fine-tuned model...")
    finetuned_model, finetuned_tokenizer = load_model(
        finetuned_model_path, use_lora=True
    )

    # Test each prompt
    for i, prompt in enumerate(test_prompts, 1):
        print("\n" + "=" * 80)
        print(f"TEST {i}/{len(test_prompts)}")
        print("=" * 80)
        print(f"\n📝 PROMPT: {prompt}\n")

        # Generate with base model
        print("🔹 Generating with BASE model...")
        base_story = generate_story(base_model, base_tokenizer, prompt)

        print("\n" + "-" * 80)
        print("BASE MODEL OUTPUT:")
        print("-" * 80)
        print(base_story)
        print("-" * 80)

        # Generate with fine-tuned model
        print("\n🔸 Generating with FINE-TUNED model...")
        finetuned_story = generate_story(finetuned_model, finetuned_tokenizer, prompt)

        print("\n" + "-" * 80)
        print("FINE-TUNED MODEL OUTPUT:")
        print("-" * 80)
        print(finetuned_story)
        print("-" * 80)

        # Wait for user
        if i < len(test_prompts):
            input("\n⏸️  Press Enter to continue to next prompt...")

    print("\n" + "=" * 80)
    print("✅ Comparison complete!")
    print("=" * 80)


def interactive_mode(model_path):
    """Interactive story generation"""
    print("\n" + "=" * 80)
    print("🎮 Interactive Story Generation Mode")
    print("=" * 80)
    print("\nType your prompts and watch StoryForge create stories!")
    print("Type 'quit' to exit\n")

    # Load model
    model, tokenizer = load_model(model_path, use_lora=True)

    while True:
        # Get prompt from user
        prompt = input("\n📝 Your prompt: ").strip()

        if prompt.lower() in ["quit", "exit", "q"]:
            print("\n👋 Thanks for using StoryForge!")
            break

        if not prompt:
            print("⚠️  Please enter a prompt!")
            continue

        # Generate story
        print("\n✍️  Generating story...\n")
        story = generate_story(model, tokenizer, prompt, max_new_tokens=800)

        print("=" * 80)
        print("📖 GENERATED STORY:")
        print("=" * 80)
        print(story)
        print("=" * 80)


def main():
    """Main execution"""
    print(
        """
    ╔════════════════════════════════════════════╗
    ║        StoryForge Model Testing           ║
    ║                                            ║
    ║  Test and compare your fine-tuned model   ║
    ╚════════════════════════════════════════════╝
    """
    )

    # Configuration
    base_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    finetuned_model_path = "models/checkpoints/base_model"

    # Check if fine-tuned model exists
    if not Path(finetuned_model_path).exists():
        print(f"❌ Fine-tuned model not found at {finetuned_model_path}")
        print("   Run finetune_base.py first!")
        return

    # Test prompts
    test_prompts = [
        "Write a short mystery story about a detective who discovers something unexpected.",
        "Tell a story about a character who finds a mysterious door in their basement.",
        "Write the opening of a sci-fi story set on a distant planet.",
        "Create a story about an unlikely friendship between two very different characters.",
    ]

    # Menu
    print("\n📋 Choose mode:")
    print("   1. Compare base vs fine-tuned model")
    print("   2. Interactive generation (fine-tuned only)")
    print("   3. Quick test (fine-tuned only)")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        compare_models(base_model_path, finetuned_model_path, test_prompts)
    elif choice == "2":
        interactive_mode(finetuned_model_path)
    elif choice == "3":
        print("\n🚀 Running quick test...")
        model, tokenizer = load_model(finetuned_model_path, use_lora=True)

        prompt = "Write a short story about a time traveler who accidentally changes history."
        print(f"\n📝 Prompt: {prompt}\n")
        print("✍️  Generating...\n")

        story = generate_story(model, tokenizer, prompt)

        print("=" * 80)
        print("📖 GENERATED STORY:")
        print("=" * 80)
        print(story)
        print("=" * 80)
        print("\n✅ Test complete!")
    else:
        print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
