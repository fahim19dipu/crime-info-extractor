# train.py

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from config import MODEL_ID, OUTPUT_DIR, TRAIN_FILE, TEST_FILE, MAX_SEQ_LENGTH
import torch

# Load tokenizer and model with LoRA and 4-bit quantization
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, peft_config)

# Load dataset
dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "test": TEST_FILE})

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH
    )

tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])

collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template="Expected Output:"
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    args=training_args,
    data_collator=collator,
)

trainer.train()