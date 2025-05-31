# Parameter-Efficient Fine-Tuning of LLaMA 3.2 (3B) on Medical Chain-of-Thought Dataset
# Complete implementation for Kaggle Notebooks

# =============================================================================
# 1. ENVIRONMENT SETUP
# =============================================================================

import os
import gc
import torch
import wandb
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from rouge_score import rouge_scorer
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Install required packages (run this in Kaggle)
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "xformers<0.0.27" trl peft accelerate bitsandbytes
# !pip install rouge-score wandb

# =============================================================================
# 2. WEIGHTS & BIASES SETUP
# =============================================================================

# Initialize wandb (replace with your API key)
# For Kaggle, add your wandb API key in secrets
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    wandb_api_key = user_secrets.get_secret("wandb_api_key")
    wandb.login(key=wandb_api_key)
except:
    print("Please set up your wandb API key in Kaggle secrets or login manually")
    # wandb.login()  # Uncomment and run manually if needed

# Initialize wandb project
wandb.init(
    project="llama-3.2-medical-cot",
    name="llama3.2-3b-medical-finetuning",
    config={
        "model_name": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "dataset": "FreedomIntelligence/huatuo-llama-med-chinese",
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_gradient_checkpointing": True,
        "random_state": 42,
        "use_rslora": False,
        "loftq_config": None,
    }
)

# =============================================================================
# 3. MODEL LOADING
# =============================================================================

def load_model_and_tokenizer():
    """Load LLaMA 3.2 3B model with 4-bit quantization"""
    
    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=42,
        use_rslora=False,   # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    
    return model, tokenizer

print("Loading model and tokenizer...")
model, tokenizer = load_model_and_tokenizer()
print("Model loaded successfully!")

# =============================================================================
# 4. DATASET PREPARATION
# =============================================================================

def load_and_prepare_dataset():
    """Load and prepare the medical Chain-of-Thought dataset"""
    
    # Load the medical dataset
    print("Loading medical Chain-of-Thought dataset...")
    
    # Using a medical CoT dataset - replace with the specific dataset you want
    try:
        # Try loading the FreedomIntelligence dataset
        dataset = load_dataset("FreedomIntelligence/huatuo-llama-med-chinese", split="train")
        print(f"Loaded dataset with {len(dataset)} examples")
    except:
        # Fallback to a general medical QA dataset
        print("Primary dataset not available, using alternative medical dataset...")
        dataset = load_dataset("medmcqa", split="train")
        print(f"Loaded fallback dataset with {len(dataset)} examples")
    
    return dataset

def format_medical_prompt(example):
    """Format the dataset into instruction-following format with CoT"""
    
    # Create a medical Chain-of-Thought prompt format
    if "question" in example and "answer" in example:
        # For medmcqa format
        question = example["question"]
        answer = example["answer"] if isinstance(example["answer"], str) else str(example["answer"])
        
        # Create CoT format
        prompt = f"""Below is a medical question. Please provide a step-by-step analysis before giving your final answer.

### Question:
{question}

### Analysis:
<think>
Let me analyze this medical question step by step:
1. Understanding the question: {question}
2. Considering relevant medical knowledge and principles
3. Evaluating possible answers or approaches
4. Reasoning through the clinical implications
</think>

### Response:
{answer}"""
    
    elif "instruction" in example:
        # For instruction-response format
        instruction = example["instruction"]
        response = example.get("output", example.get("response", ""))
        
        prompt = f"""Below is a medical instruction. Please provide a step-by-step analysis before giving your response.

### Instruction:
{instruction}

### Analysis:
<think>
Let me approach this medical question systematically:
1. Understanding the clinical scenario
2. Applying relevant medical knowledge
3. Considering differential diagnoses or treatment options
4. Formulating an evidence-based response
</think>

### Response:
{response}"""
    
    else:
        # Generic format
        prompt = f"""Below is a medical question that requires careful analysis.

### Question:
{str(example)}

### Analysis:
<think>
Step-by-step medical reasoning:
1. Analyzing the clinical presentation
2. Considering relevant pathophysiology
3. Evaluating diagnostic or therapeutic options
4. Applying evidence-based medicine principles
</think>

### Response:
Based on the analysis above, here is the medical response."""
    
    return {"text": prompt}

def prepare_training_data(dataset, sample_size=1000):
    """Prepare and split the training data"""
    
    # Sample the dataset if it's too large
    if len(dataset) > sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
        print(f"Sampled {sample_size} examples from the dataset")
    
    # Format the dataset
    formatted_dataset = dataset.map(format_medical_prompt, remove_columns=dataset.column_names)
    
    # Split into train and validation
    train_size = len(formatted_dataset) - 100  # Reserve 100 for validation
    
    train_dataset = formatted_dataset.select(range(train_size))
    val_dataset = formatted_dataset.select(range(train_size, len(formatted_dataset)))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset

# Load and prepare the dataset
raw_dataset = load_and_prepare_dataset()
train_dataset, val_dataset = prepare_training_data(raw_dataset)

# Display a sample
print("\n=== SAMPLE TRAINING EXAMPLE ===")
print(train_dataset[0]["text"][:500] + "...")

# =============================================================================
# 5. EVALUATION SETUP
# =============================================================================

def calculate_rouge_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE-L scores for model evaluation"""
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        rouge_scores.append(score['rougeL'].fmeasure)
    
    return {
        'rouge_l_mean': np.mean(rouge_scores),
        'rouge_l_std': np.std(rouge_scores)
    }

def evaluate_model_before_training(model, tokenizer, eval_dataset, num_samples=20):
    """Evaluate model performance before fine-tuning"""
    
    print("Evaluating model before fine-tuning...")
    
    # Sample evaluation examples
    eval_samples = eval_dataset.select(range(min(num_samples, len(eval_dataset))))
    
    predictions = []
    references = []
    
    for example in eval_samples:
        # Extract the question part for inference
        text = example["text"]
        question_end = text.find("### Analysis:")
        if question_end != -1:
            input_text = text[:question_end].strip()
        else:
            input_text = text.split("### Response:")[0].strip()
        
        # Generate response
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode prediction
        prediction = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract reference answer
        reference = text.split("### Response:")[-1].strip()
        
        predictions.append(prediction.strip())
        references.append(reference.strip())
    
    # Calculate ROUGE scores
    rouge_scores = calculate_rouge_score(predictions, references)
    
    print(f"Pre-training ROUGE-L: {rouge_scores['rouge_l_mean']:.4f} ± {rouge_scores['rouge_l_std']:.4f}")
    
    # Log to wandb
    wandb.log({
        "pre_training_rouge_l_mean": rouge_scores['rouge_l_mean'],
        "pre_training_rouge_l_std": rouge_scores['rouge_l_std']
    })
    
    return rouge_scores

# Evaluate before training
pre_training_scores = evaluate_model_before_training(model, tokenizer, val_dataset)

# =============================================================================
# 6. TRAINING SETUP
# =============================================================================

def setup_training_arguments():
    """Setup training arguments for fine-tuning"""
    
    return TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
        report_to="wandb",
        run_name="llama3.2-medical-cot",
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
    )

# =============================================================================
# 7. FINE-TUNING PROCESS
# =============================================================================

def train_model(model, tokenizer, train_dataset, val_dataset):
    """Fine-tune the model using SFTTrainer"""
    
    print("Starting fine-tuning process...")
    
    # Setup training arguments
    training_args = setup_training_arguments()
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    
    # Add callback for memory monitoring
    class MemoryCallback:
        def on_step_end(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_cached = torch.cuda.memory_reserved() / 1e9
                wandb.log({
                    "gpu_memory_allocated_gb": memory_used,
                    "gpu_memory_cached_gb": memory_cached,
                    "step": state.global_step
                })
    
    trainer.add_callback(MemoryCallback())
    
    # Start training
    trainer.train()
    
    print("Fine-tuning completed!")
    return trainer

# Start training
trainer = train_model(model, tokenizer, train_dataset, val_dataset)

# =============================================================================
# 8. POST-TRAINING EVALUATION
# =============================================================================

def evaluate_model_after_training(model, tokenizer, eval_dataset, num_samples=20):
    """Evaluate model performance after fine-tuning"""
    
    print("Evaluating model after fine-tuning...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Sample evaluation examples
    eval_samples = eval_dataset.select(range(min(num_samples, len(eval_dataset))))
    
    predictions = []
    references = []
    
    for example in eval_samples:
        # Extract the question part for inference
        text = example["text"]
        question_end = text.find("### Analysis:")
        if question_end != -1:
            input_text = text[:question_end].strip()
        else:
            input_text = text.split("### Response:")[0].strip()
        
        # Generate response
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode prediction
        prediction = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract reference answer
        reference = text.split("### Response:")[-1].strip()
        
        predictions.append(prediction.strip())
        references.append(reference.strip())
    
    # Calculate ROUGE scores
    rouge_scores = calculate_rouge_score(predictions, references)
    
    print(f"Post-training ROUGE-L: {rouge_scores['rouge_l_mean']:.4f} ± {rouge_scores['rouge_l_std']:.4f}")
    
    # Log to wandb
    wandb.log({
        "post_training_rouge_l_mean": rouge_scores['rouge_l_mean'],
        "post_training_rouge_l_std": rouge_scores['rouge_l_std']
    })
    
    return rouge_scores, predictions, references

# Evaluate after training
post_training_scores, sample_predictions, sample_references = evaluate_model_after_training(
    model, tokenizer, val_dataset
)

# Compare scores
print("\n=== ROUGE-L SCORE COMPARISON ===")
print(f"Pre-training:  {pre_training_scores['rouge_l_mean']:.4f} ± {pre_training_scores['rouge_l_std']:.4f}")
print(f"Post-training: {post_training_scores['rouge_l_mean']:.4f} ± {post_training_scores['rouge_l_std']:.4f}")
improvement = post_training_scores['rouge_l_mean'] - pre_training_scores['rouge_l_mean']
print(f"Improvement:   {improvement:.4f}")

# Log comparison to wandb
wandb.log({
    "rouge_l_improvement": improvement,
    "final_comparison": {
        "pre_training": pre_training_scores['rouge_l_mean'],
        "post_training": post_training_scores['rouge_l_mean']
    }
})

# =============================================================================
# 9. MODEL SAVING AND HUGGING FACE UPLOAD
# =============================================================================

def save_and_upload_model(model, tokenizer, model_name="llama-3.2-3b-medical-cot"):
    """Save the fine-tuned model and upload to Hugging Face"""
    
    print("Saving fine-tuned model...")
    
    # Save LoRA adapter
    model.save_pretrained(f"lora_model_{model_name}")
    tokenizer.save_pretrained(f"lora_model_{model_name}")
    
    print(f"Model saved to lora_model_{model_name}/")
    
    # Optional: Upload to Hugging Face Hub
    # Uncomment and configure the following if you want to upload to HF
    """
    from huggingface_hub import HfApi, login
    
    # Login to Hugging Face (set your token in Kaggle secrets)
    try:
        hf_token = user_secrets.get_secret("hf_token")
        login(token=hf_token)
        
        # Upload model
        model.push_to_hub(
            f"your-username/{model_name}",
            token=hf_token,
            private=False
        )
        
        tokenizer.push_to_hub(
            f"your-username/{model_name}",
            token=hf_token,
            private=False
        )
        
        print(f"Model uploaded to https://huggingface.co/your-username/{model_name}")
        
    except Exception as e:
        print(f"Upload failed: {e}")
        print("Please set up your Hugging Face token in Kaggle secrets to enable upload")
    """

# Save the model
save_and_upload_model(model, tokenizer)

# =============================================================================
# 10. INFERENCE DEMONSTRATION
# =============================================================================

def demonstrate_inference(model, tokenizer):
    """Demonstrate inference with the fine-tuned model"""
    
    print("\n=== INFERENCE DEMONSTRATION ===")
    
    # Sample medical questions for testing
    test_questions = [
        "What are the primary symptoms and treatment options for Type 2 diabetes?",
        "Explain the pathophysiology of myocardial infarction and its immediate management.",
        "What are the differential diagnoses for acute abdominal pain in a 35-year-old patient?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test Question {i} ---")
        print(f"Question: {question}")
        
        # Format prompt
        prompt = f"""Below is a medical question. Please provide a step-by-step analysis before giving your final answer.

### Question:
{question}

### Analysis:
<think>"""
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode and display response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = full_response[len(prompt):].strip()
        
        print(f"Generated Response:\n{generated_part}")
        print("-" * 80)

# Demonstrate inference
demonstrate_inference(model, tokenizer)

# =============================================================================
# 11. FINAL SUMMARY AND INSTRUCTIONS
# =============================================================================

print("\n" + "="*80)
print("FINE-TUNING COMPLETED SUCCESSFULLY!")
print("="*80)

print(f"""
SUMMARY:
- Model: LLaMA 3.2 (3B) with LoRA fine-tuning
- Dataset: Medical Chain-of-Thought 
- Training samples: {len(train_dataset)}
- Validation samples: {len(val_dataset)}
- Pre-training ROUGE-L: {pre_training_scores['rouge_l_mean']:.4f}
- Post-training ROUGE-L: {post_training_scores['rouge_l_mean']:.4f}
- Improvement: {post_training_scores['rouge_l_mean'] - pre_training_scores['rouge_l_mean']:.4f}

FILES SAVED:
- LoRA adapter: lora_model_llama-3.2-3b-medical-cot/
- Tokenizer: lora_model_llama-3.2-3b-medical-cot/

USAGE INSTRUCTIONS:
To use the fine-tuned model in a new session:

1. Load the base model:
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
       max_seq_length=2048,
       dtype=None,
       load_in_4bit=True,
   )

2. Load the LoRA adapter:
   model = FastLanguageModel.get_peft_model(model, ...)  # Apply LoRA config
   model.load_adapter("lora_model_llama-3.2-3b-medical-cot")

3. Generate medical responses:
   inputs = tokenizer("Your medical question here", return_tensors="pt")
   outputs = model.generate(**inputs, max_new_tokens=256)
   response = tokenizer.decode(outputs[0], skip_special_tokens=True)

WANDB LOGS: Check your wandb dashboard for detailed training metrics.
""")

# Clean up memory
torch.cuda.empty_cache()
gc.collect()

# Finish wandb run
wandb.finish()

print("\nSetup complete! The model is ready for medical Chain-of-Thought inference.")