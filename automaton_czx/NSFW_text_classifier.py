# DistilBERT NSFW text classifier (michellejieli/NSFW_text_classifier)
# Local copy used by main_czx_classifier.py

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_nsfw_classifier():
    tokenizer = AutoTokenizer.from_pretrained("michellejieli/NSFW_text_classifier", token=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "michellejieli/NSFW_text_classifier",
        token=True
    ).to(device)

    classifier = pipeline(
        "text-classification",
        tokenizer=tokenizer,
        model=model,
        device=0 if device.type == "cuda" else -1,
        batch_size=8,
        truncation=True
    )
    return classifier

classifier = load_nsfw_classifier()

def text_classifier_check(text_inputs):
    output = classifier(text_inputs)
    if output[0]['label'] == 'NSFW':
        is_safe = False
    else:
        is_safe = True
    return is_safe

if __name__ == "__main__":
    input_file = "../results/OptJail_prompt.txt"
    with open(input_file, 'r', encoding='utf-8') as f:
        origin_prompts = [line.strip() for line in f if line.strip()]

    success_count = 0

    for prompt in tqdm(origin_prompts, desc="Testing text safety bypass rate"):
        is_safe = text_classifier_check(prompt)
        if is_safe:
            success_count += 1

    print(f"Success: {success_count}/{len(origin_prompts)}")
