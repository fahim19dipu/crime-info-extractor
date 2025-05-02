# Crime Info Extractor with Titulm-Gemma

This repository contains the full pipeline to fine-tune the `titulm-gemma-2-2b-v1.1` LLM to extract structured information from unstructured crime news articles.

## 🧠 Model Task
The model reads raw news text and extracts:
- 📅 Date & Time
- 📍 Location
- 🚨 Crime Type
- 💰 Property Loss
- 👤 Number of Victims

## 🛠️ Technologies Used
- 🤗 Transformers
- 🧩 TRL (SFTTrainer)
- 🔧 PEFT (LoRA)
- 🧮 Bitsandbytes (4-bit quantization)
- ⚡ PyTorch + Accelerate

## 🗂️ Project Structure
- `dataset.json` – Raw annotated news articles
- `train.jsonl`, `test.jsonl` – Prompt-formatted SFT training data
- `train.py` – Fine-tuning pipeline
- `inference.py` – Generate structured info from new articles
- `config.py` – Model and training constants
-`data_prep.py` – Dataset formatting and train/test splitting-
- `crime-extractor-lora/` – Fine-tuned model directory (not committed)

## 🚀 Inference Example

```python
extract_crime_info("On April 10, 2023, a robbery occurred in downtown Chicago...")
```
## Output
```python
Date time: April 10, 2023
Location: downtown Chicago
Crime Type: Robbery
Property Loss: $5000
Number of Victim: 1
```
