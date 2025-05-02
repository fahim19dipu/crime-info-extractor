# data_prep.py

import json
import random

def format_dataset(input_file, train_file, test_file, split_ratio=0.8):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.shuffle(data)
    formatted_data = []

    for item in data:
        prompt = (
            "Extract the following details from the article:\n"
            "Crime Date time, Crime Location, Crime Type, Property Loss, Number of Victim.\n\n"
            f"Article:\n{item['text']}\n\n"
            "Expected Output:\n"
            f"Date time: {item['label']['Date time']}\n"
            f"Location: {item['label']['Location']}\n"
            f"Type: {item['label']['Crime Type']}\n"
            f"Property Loss: {item['label']['Property Loss']}\n"
            f"Number of Victim: {item['label']['Number of Victim']}\n"
        )
        formatted_data.append({"text": prompt})

    split_index = int(split_ratio * len(formatted_data))
    with open(train_file, "w", encoding="utf-8") as f:
        for entry in formatted_data[:split_index]:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(test_file, "w", encoding="utf-8") as f:
        for entry in formatted_data[split_index:]:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {split_index} train and {len(formatted_data) - split_index} test samples.")