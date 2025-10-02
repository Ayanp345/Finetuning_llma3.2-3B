# Finetuning_llma3.2-3B
This repository showcases end-to-end supervised fine-tuning of the Llama 3.2B model using Unsloth, Hugging Face’s Transformers, and TRL libraries



Description
This project demonstrates how to perform supervised fine-tuning of the Llama 3.2B language model using the Unsloth library. It covers every step required—from installing necessary libraries and loading the model/dataset, to applying chat templates, formatting data, training with parameter-efficient LoRA techniques, and running inference. The tutorial is tailored for practical use on available GPU resources and emphasizes reproducibility for ML hackathons or personal research.


Project Overview
Model: Llama 3.2B (Instruct model)

Goal: Adapt a pre-trained LLM to follow task-specific instructions using a curated dataset.

Techniques:

LoRA (Low-Rank Adaptation) for parameter-efficient finetuning

Quantization (4-bit) to minimize GPU memory use

Instruction-tuning format using chat templates

Workflow
1. Installation and Setup
python
pip install unsloth transformers trl
Set up a GPU runtime (e.g., T4/Google Colab).

2. Preparing Model and Tokenizer
python
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

model_name = "unsloth/Llama-3-2-1B-Instruct"
max_seq_length = 2048

model = FastLanguageModel.from_pretrained(model_name, load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
3. Load and Format the Dataset
Use an instruction dataset (like FineTome-100k).

Standardize to a ShareGPT-style chat format.

Apply a chat template to convert message dicts to a single training string.

python
def formatting_func(example):
    return tokenizer.apply_chat_template(example["conversations"], tokenize=False)

# Apply formatting, keeping only the relevant "text" field.
dataset = dataset.map(lambda x: {"text": formatting_func(x)}, remove_columns=["other_columns"])
4. Initialize SFTTrainer and Training Arguments
python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    formatting_func=formatting_func,
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
    ),
)
5. Train and Save Model
python
trainer.train()
trainer.save_model("finetuned-llama3.2b")
6. Inference Example
python
prompt = "What are the key principles of investment?"
formatted_prompt = tokenizer.apply_chat_template([{"role":"user", "content":prompt}], tokenize=False)
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
print(response)
Key Takeaways
Use LoRA and 4-bit quantization to fine-tune large models on limited hardware.

Standardize dataset dialogs into a single string using chat templates before training.

Include a required formatting_func in Unsloth SFTTrainer for dataset processing.

Save and checkpoint models for reusability across sessions.

This workflow enables efficient, reproducible, and memory-conscious fine-tuning of LLMs for custom instruction-following tasks, suitable for research, hackathons, and rapid prototyping
