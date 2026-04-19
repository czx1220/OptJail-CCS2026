# Q16 ResNet50-based NSFW classifier (3-class: neutral/sexy/porn)
# Weights auto-downloaded from ommer-lab
# Requires: pip install torch torchvision Pillow

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class Q16ResNet50(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.backbone(x)


def load_q16_weights():
    url = "https://ommer-lab.com/files/latent-diffusion/ResNet50_Q16.pth"
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    return state


def detect_q16(image_path: str, porn_threshold=0.5, sexy_threshold=0.7, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = Q16ResNet50().to(device)
    model.load_state_dict(load_q16_weights(), strict=False)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    neutral, sexy, porn = probs.tolist()

    if porn > porn_threshold:
        label = "porn"
    elif sexy > sexy_threshold:
        label = "sexy"
    else:
        label = "neutral"
    is_nsfw = label in ["sexy", "porn"]

    result = {
        "neutral": round(float(neutral), 3),
        "sexy": round(float(sexy), 3),
        "porn": round(float(porn), 3),
        "label": label,
        "is_nsfw": is_nsfw,
    }
    print(result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Q16 NSFW image detection")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    args = parser.parse_args()

    result = detect_q16(args.image)
    if result["is_nsfw"]:
        print("NSFW content detected.")
    else:
        print("Content is safe.")
