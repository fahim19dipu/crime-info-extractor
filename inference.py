# inference.py

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import OUTPUT_DIR
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

def extract_crime_info(article, max_new_tokens=200):
    prompt = (
        "Extract the following details from the article:\n"
        "Crime Date time, Crime Location, Crime Type, Property Loss, Number of Victim.\n\n"
        f"Article:\n{article}\n\n"
        "Expected Output:\n"
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()