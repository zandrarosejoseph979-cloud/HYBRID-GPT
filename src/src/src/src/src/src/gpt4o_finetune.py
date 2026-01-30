"""
GPT-4o Fine-tuning Script (API-based)
This script prepares image-text pairs and sends them to OpenAI fine-tuning endpoints.
"""

import json

def prepare_jsonl(image_path, label):
    sample = {
        "messages": [
            {"role": "user", "content": [{"type": "input_text", "text": "Classify this product image."},
                                          {"type": "input_image", "image_path": image_path}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": label}]}
        ]
    }
    return sample

with open("data/train.jsonl", "w") as f:
    f.write(json.dumps(prepare_jsonl("image1.jpg", "Footwear")) + "\n")

print("GPT-4o fine-tuning dataset prepared.")
