# 🎭 StoryForge: Constitutional AI for Creative Writing

A creative writing assistant that generates stories while respecting nuanced creative constraints using Constitutional AI and RLHF.

## 🎯 Project Goals

Build three versions of a story generation model:
1. **Base Model**: Llama 3.1 8B fine-tuned on creative writing
2. **Constitutional AI**: Self-critiquing system that improves story quality
3. **RLHF-Enhanced**: Further refined using human preference feedback

## 🏗️ Project Structure

```
storyforge/
├── data/                    # Datasets
│   ├── raw/                # Original datasets
│   ├── processed/          # Cleaned and formatted data
│   └── preferences/        # Human preference rankings
├── models/                  # Model checkpoints
│   ├── checkpoints/        # Base fine-tuned model
│   ├── constitutional/     # Constitutional AI model
│   └── rlhf/              # RLHF-trained model
├── scripts/                 # Training and evaluation scripts
│   ├── training/           # Fine-tuning scripts
│   ├── evaluation/         # Model evaluation
│   ├── constitutional/     # Constitutional AI logic
│   └── rlhf/              # RLHF implementation
├── configs/                 # Configuration files
├── notebooks/               # Jupyter notebooks for exploration
├── outputs/                 # Generated stories and metrics
└── demo/                    # Streamlit demo app
```

## 🚀 Quick Start

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Phase 1: Base Model Fine-tuning
```bash
python scripts/training/data_prep.py
python scripts/training/finetune_base.py --config configs/base_model.yaml
```

### Phase 2: Constitutional AI
```bash
python scripts/constitutional/critique_and_revise.py
```

### Phase 3: RLHF Training
```bash
python scripts/rlhf/train_reward_model.py
python scripts/rlhf/ppo_training.py
```

## 📊 Progress Tracker

- [ ] **Week 1-2**: Base model fine-tuning
  - [x] Environment setup
  - [ ] Data collection and preprocessing
  - [ ] LoRA fine-tuning on creative writing
  - [ ] Baseline evaluation
  
- [ ] **Week 3-4**: Constitutional AI
  - [ ] Define creative writing constitution
  - [ ] Implement critique-revision loop
  - [ ] Evaluate improvements
  
- [ ] **Week 5-6**: RLHF
  - [ ] Collect preference data
  - [ ] Train reward model
  - [ ] PPO fine-tuning
  - [ ] Final evaluation
  
- [ ] **Week 7**: Demo & Documentation
  - [ ] Build Streamlit demo
  - [ ] Write technical blog post
  - [ ] Create comparison visualizations
  - [ ] Prepare for release

## 🛠️ Tech Stack

- **Base Framework**: PyTorch
- **Model**: Llama 3.1 8B (Meta)
- **Fine-tuning**: PEFT (LoRA/QLoRA)
- **RLHF**: TRL library
- **Tracking**: Weights & Biases
- **Demo**: Streamlit

## 📚 Key Concepts

### LoRA (Low-Rank Adaptation)
Efficient fine-tuning by training only small adapter layers instead of the full model.

### Constitutional AI
Models critique and revise their own outputs based on predefined principles.

### RLHF (Reinforcement Learning from Human Feedback)
Using human preferences to train a reward model, then optimizing the language model with RL.

## 📖 Learning Resources

- [Constitutional AI Paper](https://arxiv.org/abs/2212.08073)
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT Docs](https://huggingface.co/docs/peft)

## 🎓 What I Learned

*[Keep a running log of key insights and challenges]*

---

**Status**: 🚧 In Progress | **Started**: [Date] | **Target Completion**: 5 weeks