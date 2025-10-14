# 🚀 StoryForge Week 1: Complete Action Plan

## ✅ Days 1-2: COMPLETED
- [x] Environment setup
- [x] Project structure created
- [x] Llama 3.1 loading verified
- [x] LoRA concepts understood

---

## 📅 Day 3: Data Collection (4-6 hours)

### Morning Session (2-3 hours)
**Goal:** Collect 50k+ creative writing samples

1. **Run data collection script**
   ```bash
   cd ~/storyforge
   source venv/bin/activate
   python scripts/training/data_prep.py  # Save as collect_writing_data.py
   ```

2. **What to expect:**
   - Script will download WritingPrompts dataset (~30k stories)
   - TinyStories dataset (~20k stories)
   - Takes 15-20 minutes depending on internet speed
   - Total size: ~300-500MB

3. **Verify collection:**
   ```bash
   ls -lh data/raw/
   # Should see: writingprompts.jsonl, tinystories.jsonl, collection_report.json
   ```

4. **Review the data:**
   - Open `data/raw/collection_report.json`
   - Check the statistics
   - Read a few sample stories to understand quality

### Afternoon Session (2-3 hours)
**Goal:** Clean and prepare data for training

1. **Run preprocessing script**
   ```bash
   python scripts/training/preprocess_data.py
   ```

2. **What happens:**
   - Text cleaning and normalization
   - Quality filtering (removes spam, too short, corrupted)
   - Train/validation split (95%/5%)
   - Formatting for Llama 3.1 instruction format
   - Takes 10-15 minutes

3. **Verify preprocessing:**
   ```bash
   ls -lh data/processed/
   # Should see: train.jsonl, val.jsonl, metadata.json
   ```

4. **Inspect formatted data:**
   ```bash
   head -n 1 data/processed/train.jsonl | python -m json.tool
   ```
   - Look at how the instruction format works
   - Understand the structure

### Evening: Learning & Documentation (1 hour)

1. **Read these papers (skim, don't deep dive):**
   - LoRA paper abstract: https://arxiv.org/abs/2106.09685
   - Llama 2 paper (section on instruction tuning): https://arxiv.org/abs/2307.09288

2. **Update your learning log:**
   ```markdown
   # Day 3: Data Collection
   
   ## What I Did:
   - Collected 50k stories from WritingPrompts and TinyStories
   - Preprocessed and cleaned data
   - Split into train/val sets
   
   ## Key Insights:
   - Data quality matters more than quantity
   - Instruction format is crucial for chat models
   - [Your observations here]
   
   ## Questions:
   - [Any questions you have]
   ```

---

## 📅 Day 4: Understanding Training (3-4 hours)

### Morning: Code Review (2 hours)

**Don't run training yet!** First, understand the code.

1. **Open `finetune_base.py`**

2. **Study these sections carefully:**
   - `ModelConfig` class: What does each parameter do?
   - `load_model_and_tokenizer()`: How is quantization applied?
   - `setup_lora()`: How are LoRA adapters added?
   - `load_datasets()`: How is data tokenized?

3. **Answer these questions (write them down):**
   - Why do we use 4-bit quantization?
   - What does `gradient_accumulation_steps=8` mean?
   - Why is `lora_r=16` and not 64 or 4?
   - What's the effective batch size?
   - How many trainable parameters will we have?

### Afternoon: Set Up Weights & Biases (1 hour)

W&B is essential for tracking experiments!

1. **Create W&B account:**
   - Go to https://wandb.ai
   - Sign up (free account)
   - Get your API key

2. **Login from terminal:**
   ```bash
   wandb login
   # Paste your API key when prompted
   ```

3. **Test W&B:**
   ```python
   import wandb
   wandb.init(project="storyforge-test")
   wandb.log({"test": 1})
   wandb.finish()
   ```

4. **Check W&B dashboard:**
   - You should see the test run
   - Explore the interface

### Evening: Calculate Training Time (1 hour)

**Important:** Estimate how long training will take!

1. **Your hardware:**
   - Mac M4 with 24GB RAM
   - Batch size: 2
   - Gradient accumulation: 8
   - ~40k training samples
   - 3 epochs

2. **Rough estimates:**
   - On Mac M4: ~30-40 hours total for 3 epochs
   - This is SLOW but doable
   - Consider using Google Colab for actual training

3. **Colab setup (recommended):**
   - Open Google Colab
   - Change runtime to GPU (T4 is free)
   - Upload your data to Google Drive
   - Modify paths in script to read from Drive
   - On Colab GPU: ~4-6 hours for 3 epochs

4. **Decision time:**
   - Mac: Good for testing, slow for full training
   - Colab: Free GPU, faster, but session limits (12 hours)
   - Recommendation: Start on Mac, move to Colab if too slow

---

## 📅 Day 5: Start Training! (Setup: 1 hour, Training: 4-6 hours on Colab)

### Option A: Local Mac Training

```bash
cd ~/storyforge
source venv/bin/activate

# Modify config for faster iteration (optional - for testing)
# In finetune_base.py, change:
# - num_train_epochs: 1 (instead of 3)
# - Use subset of data for quick test

python scripts/training/finetune_base.py
```

**Monitor training:**
- Watch W&B dashboard for metrics
- Check loss curves
- Monitor memory usage with Activity Monitor

### Option B: Google Colab Training (RECOMMENDED)

1. **Set up Colab notebook:**
   ```python
   # Install dependencies
   !pip install transformers accelerate peft bitsandbytes datasets wandb
   
   # Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Copy your data to Colab
   !cp -r /content/drive/MyDrive/storyforge/data .
   !cp /content/drive/MyDrive/storyforge/scripts/training/finetune_base.py .
   
   # Login to W&B
   import wandb
   wandb.login()
   
   # Run training
   !python finetune_base.py
   ```

2. **What to watch:**
   - Training loss should decrease
   - Validation loss should follow training loss
   - If validation loss increases while training decreases: overfitting!
   - Target: validation loss < 2.0 after 3 epochs

3. **Save checkpoints:**
   - Training saves every 500 steps
   - Copy best checkpoint to your Drive

### During Training: Learn More (2-3 hours)

While training runs, don't just wait! Study:

1. **Watch this video:** 
   - "Fine-tuning LLMs with LoRA" by Weights & Biases (YouTube)
   - ~30 minutes

2. **Read Anthropic's Constitutional AI blog:**
   - https://www.anthropic.com/index/constitutional-ai-harmlessness-from-ai-feedback
   - This is what you'll implement next week!

3. **Explore W&B dashboard:**
   - Understand each metric
   - Compare different runs
   - Learn to spot problems in training curves

---

## 📅 Day 6: Evaluate Your Model (4-5 hours)

### Morning: Test Generation (2 hours)

Training should be done by now!

1. **Run test script:**
   ```bash
   python scripts/evaluation/test_model.py
   ```
   - Choose option 3 (quick test) first
   - Then try interactive mode (option 2)

2. **Generate stories with different prompts:**
   ```
   Test these:
   - "Write a mystery story about..."
   - "Tell a sci-fi tale where..."
   - "Create a romance story featuring..."
   - "Write a horror story in which..."
   ```

3. **Evaluate quality manually:**
   - Is the story coherent?
   - Does it follow the prompt?
   - Is it creative or generic?
   - Any repetition or nonsense?

### Afternoon: Compare Models (2 hours)

1. **Run comparison mode:**
   ```bash
   python scripts/evaluation/test_model.py
   # Choose option 1
   ```

2. **Analyze differences:**
   - Base model vs your fine-tuned model
   - Which is more creative?
   - Which follows prompts better?
   - Take screenshots for your blog post!

3. **Create comparison document:**
   ```markdown
   # Model Comparison Results
   
   ## Prompt 1: [prompt]
   ### Base Model:
   [output]
   
   ### Fine-tuned Model:
   [output]
   
   ### Analysis:
   [Your thoughts]
   ```

### Evening: Metrics & Analysis (1 hour)

1. **Review W&B metrics:**
   - Final training loss
   - Final validation loss
   - Training time
   - GPU utilization

2. **Calculate improvements:**
   - How much did loss decrease?
   - Perplexity improvement?

3. **Document everything:**
   - Update your README with results
   - Add loss curves (download from W&B)
   - Write up initial findings

---

## 📅 Day 7: Documentation & Prep for Week 2 (4-5 hours)

### Morning: Create Demo (2-3 hours)

Build a simple Streamlit app to showcase your model:

```python
# demo/app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

st.title("🎭 StoryForge: AI Creative Writer")

# [Add model loading code here]
# [Add generation interface]
# [Add settings sliders: temperature, max length]
```

Test locally:
```bash
streamlit run demo/app.py
```

### Afternoon: Week 1 Wrap-up (2 hours)

1. **Update your project tracker:**
   - Mark Week 1 tasks as complete
   - Document what worked and what didn't
   - Note any blockers

2. **Write a blog post draft:**
   ```markdown
   # Fine-tuning Llama 3.1 for Creative Writing with LoRA
   
   ## Introduction
   [Why you did this, the goal]
   
   ## Methodology
   [Data collection, LoRA setup, training]
   
   ## Results
   [Comparisons, examples, metrics]
   
   ## Lessons Learned
   [Key insights]
   
   ## What's Next
   [Constitutional AI preview]
   ```

3. **Commit everything to GitHub:**
   ```bash
   git add .
   git commit -m "Week 1: Base model fine-tuning complete"
   git push
   ```

4. **Share progress:**
   - Post on Twitter/LinkedIn with:
     - Side-by-side story comparisons
     - Loss curves from W&B
     - Link to your GitHub
     - Use hashtags: #AI #MachineLearning #LLM

### Evening: Prep for Week 2 (1 hour)

1. **Read Constitutional AI paper:**
   - https://arxiv.org/abs/2212.08073
   - Focus on methodology section

2. **Plan Week 2 approach:**
   - What principles for creative writing?
   - How to implement critique loop?
   - What tools needed?

3. **Set up Week 2 environment:**
   - Install Anthropic API client: `pip install anthropic`
   - Get API key from anthropic.com
   - Create `scripts/constitutional/` directory structure

---

## 📊 Success Metrics for Week 1

By end of Week 1, you should have:

- [x] 50k+ stories collected and processed
- [x] Fine-tuned Llama 3.1 8B with LoRA
- [x] Validation loss < 2.5
- [x] Model generates coherent stories
- [x] Working demo app
- [x] Documentation on GitHub
- [x] Blog post draft
- [x] W&B dashboard with all metrics

## 🚨 Common Issues & Solutions

### Issue: Out of memory during training
**Solution:** 
- Reduce batch size to 1
- Increase gradient accumulation steps to 16
- Use Colab instead of Mac

### Issue: Model generates repetitive text
**Solution:**
- Add temperature > 0.7
- Use top_p sampling
- Check if training loss is too low (overfitting)

### Issue: Training too slow on Mac
**Solution:**
- Switch to Google Colab (free GPU)
- Or reduce dataset size for faster iteration
- Focus on quality over speed for first run

### Issue: Model doesn't follow prompts
**Solution:**
- Check instruction formatting
- Train for more epochs
- Verify prompt structure in test script

---

## 💡 Pro Tips

1. **Don't wait for perfection:** Your first model won't be perfect. That's okay! Iterate.

2. **Document everything:** Future you will thank present you for good notes.

3. **Save checkpoints:** Training can crash. Save frequently.

4. **Share progress:** Post updates even if incomplete. It creates accountability.

5. **Ask for help:** Stuck? Come back to Claude with specific questions.

---

## 🎯 Week 2 Preview

Next week you'll implement Constitutional AI:
- Define creative writing principles
- Build critique-revision loop  
- Use Claude API as judge
- Compare base vs constitutional versions

This is where it gets REALLY interesting and impressive to hiring managers!

---

**You've got this! 🚀 Let's build something amazing!**