"""
Test script to load Llama 3.1 8B and generate text
Run this to verify your setup works!
macOS compatible version (no bitsandbytes)
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_llama_generation():
    """
    Load Llama 3.1 8B and test generation
    macOS version using MPS (Metal Performance Shaders)
    """

    print("🚀 Loading Llama 3.1 8B...")

    # Check available device
    if torch.backends.mps.is_available():
        # device = "mps"
        print("💾 Using device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        # device = "cuda"
        print("💾 Using device: CUDA (GPU)")
    else:
        # device = "cpu"
        print("💾 Using device: CPU")

    # Model name - we'll use the instruct version
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Note: You'll need to request access to Llama on HuggingFace first!

    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    print("📥 Loading model (this might take 2-3 minutes)...")
    print("⚠️  Note: Without quantization, this uses ~16GB RAM")
    start_time = time.time()

    # Load model with float16 precision (reduces memory vs float32)
    # This is the best option for macOS without bitsandbytes
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Half precision - reduces memory
        device_map="auto",  # Automatically handles device placement
        low_cpu_mem_usage=True,  # Reduces RAM usage during loading
    )

    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f} seconds!")

    # Test generation
    print("\n" + "=" * 50)
    print("🎭 Testing Story Generation")
    print("=" * 50 + "\n")

    # Creative writing prompt
    prompt = """Write the opening paragraph of a mystery story.
The story should hook the reader immediately."""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("✍️  Generating story...")
    generation_start = time.time()

    # Generate text
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.8,  # Higher = more creative
        top_p=0.9,  # Nucleus sampling
        do_sample=True,  # Enable sampling for creativity
        pad_token_id=tokenizer.eos_token_id,
    )

    generation_time = time.time() - generation_start

    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n📖 Generated in {generation_time:.2f} seconds:\n")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

    # Model info
    print("\n📊 Model Information:")
    print(f"   • Parameters: {model.num_parameters() / 1e9:.2f}B")
    print(f"   • Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
    print(f"   • Device: {model.device}")

    return model, tokenizer


def test_batch_generation(model, tokenizer):
    """
    Test generating multiple story variants at once
    """
    print("\n" + "=" * 50)
    print("🎲 Testing Batch Generation")
    print("=" * 50 + "\n")

    prompts = [
        "Once upon a time in a dystopian future,",
        "The old mansion stood alone on the hill,",
        "She opened the mysterious package and gasped,",
    ]

    # Tokenize all prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        model.device
    )

    print(f"✍️  Generating {len(prompts)} stories in parallel...")

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.9,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
    )

    # Decode each
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"\n📖 Story {i + 1}:")
        print("-" * 50)
        print(text)
        print("-" * 50)


if __name__ == "__main__":
    print(
        """
    ╔════════════════════════════════════════╗
    ║   StoryForge - Llama 3.1 Test Suite   ║
    ║              (macOS Version)           ║
    ║   Testing model loading and generation ║
    ╚════════════════════════════════════════╝
    """
    )

    # Important note about access
    print("⚠️  IMPORTANT: Before running this script:")
    print("   1. Go to https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
    print("   2. Request access (usually approved in minutes)")
    print("   3. Run: huggingface-cli login")
    print("   4. Enter your HuggingFace token\n")

    print("💡 macOS Notes:")
    print("   • No quantization available (bitsandbytes not supported)")
    print("   • Model will use ~16GB RAM instead of ~6GB")
    print("   • Generation may be slower than on CUDA GPUs\n")

    input("Press Enter to continue...")

    try:
        # Test basic generation
        model, tokenizer = test_llama_generation()

        # Test batch generation
        test_batch_generation(model, tokenizer)

        print("\n✅ All tests passed! Your environment is ready!")
        print("💡 Next step: Start collecting creative writing data!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Make sure you have HuggingFace access to Llama")
        print("   • Check your token: huggingface-cli whoami")
        print("   • Ensure you have enough RAM (needs ~16GB)")
        print("   • Try restarting your terminal")
        print("   • If MPS issues occur, the model will fall back to CPU")
