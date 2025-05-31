# 🧠 Parameter-Efficient Fine-Tuning of LLaMA 3.2 (3B) on Medical Chain-of-Thought Dataset

This project implements **parameter-efficient supervised fine-tuning** of the [LLaMA 3.2 (3B)] model using **Low-Rank Adaptation (LoRA)** on a specialized **medical Chain-of-Thought (CoT) dataset**. It leverages the [Unsloth](https://github.com/unslothai/unsloth) framework to achieve efficient fine-tuning on consumer-grade GPUs, making powerful medical AI accessible and efficient.

---

## 📚 Overview

- **Model:** LLaMA 3.2 (3B), 4-bit quantized
- **Fine-Tuning Method:** LoRA (Low-Rank Adaptation)
- **Frameworks Used:** Unsloth, Transformers, TRL, WandB
- *
- *Dataset:** Medical CoT dataset from Hugging Face
- **Platform:** Trained on Kaggle GPU (Free Tier)

---

## 🔧 Features

- 📦 Efficient 4-bit fine-tuning via LoRA
- 📈 ROUGE-L based evaluation before and after fine-tuning
- 🧪 Structured medical reasoning using Chain-of-Thought prompts
- 🔗 Deployment-ready model on Hugging Face
- 📋 Fully reproducible training pipeline

---

## 🛠️ Installation

```bash
pip install torch wandb numpy pandas datasets transformers trl unsloth rouge-score
```

---

## Project Structure
.
├── notebooks/
│   └── fine_tuning_llama_medical.ipynb
├── lora_model/
│   └── [saved_adapter_files]
├── inference.py
├── utils.py
├── README.md
└── requirements.txt

---
