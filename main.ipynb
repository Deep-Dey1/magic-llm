import torch
import evaluate
import wandb
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq, EarlyStoppingCallback
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# Step 1: Initialize Weights & Biases
wandb.init(project="mistral-qa-finetuning")

# Step 2: Load Mistral 7B Model & Tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# Step 3: Load and Verify Dataset Format
dataset = load_dataset("squad_v2")

def format_qa_sample(example):
    """Format dataset into question-context-answer format for Mistral 7B"""
    question = example["question"].strip()
    context = example["context"].strip()
    answer = example["answers"]["text"][0] if example["answers"]["text"] else "No answer"
    
    formatted_input = f"Question: {question}\nContext: {context}\nAnswer: {answer}"
    return {"text": formatted_input}

formatted_train_dataset = dataset["train"].map(format_qa_sample)
formatted_valid_dataset = dataset["validation"].map(format_qa_sample)

# Step 4: Tokenization and Preprocessing
def preprocess_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"]
    return tokenized_inputs

tokenized_train = formatted_train_dataset.map(preprocess_function, batched=True)
tokenized_valid = formatted_valid_dataset.map(preprocess_function, batched=True)

# Step 5: LoRA Configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    bias="all"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Step 6: Training Arguments
training_args = TrainingArguments(
    output_dir="./mistral_qa_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=3e-5,
    weight_decay=0.01,
    num_train_epochs=5,
    logging_steps=10,
    save_total_limit=2,
    fp16=True,
    load_best_model_at_end=True,
    push_to_hub=False
)

# Step 7: Define Evaluation Metrics
f1_metric = evaluate.load("f1")
exact_match_metric = evaluate.load("exact_match")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    f1 = f1_metric.compute(predictions=preds, references=labels)["f1"]
    exact_match = exact_match_metric.compute(predictions=preds, references=labels)["exact_match"]
    
    return {"f1": f1, "exact_match": exact_match}

# Step 8: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Step 9: Evaluate Before Fine-Tuning
baseline_results = trainer.evaluate()
wandb.log({"Baseline F1 Score": baseline_results["eval_f1"], "Baseline Exact Match": baseline_results["eval_exact_match"]})

# Step 10: Fine-Tune the Model
trainer.train()

# Step 11: Evaluate After Fine-Tuning
final_results = trainer.evaluate()
wandb.log({"Final F1 Score": final_results["eval_f1"], "Final Exact Match": final_results["eval_exact_match"]})

# Step 12: Save Fine-Tuned Model
model.save_pretrained("./mistral_qa_finetuned")
tokenizer.save_pretrained("./mistral_qa_finetuned")

# Step 13: Load and Test the Fine-Tuned Model
from transformers import pipeline

qa_pipeline = pipeline("text-generation", model="./mistral_qa_finetuned", tokenizer=tokenizer)

# Example inference
context = "Albert Einstein was a theoretical physicist who developed the theory of relativity."
question = "Who developed the theory of relativity?"
input_text = f"Question: {question}\nContext: {context}\nAnswer:"

generated_answer = qa_pipeline(input_text, max_length=100)[0]['generated_text']
print("\nGenerated Answer:", generated_answer)

wandb.finish()
