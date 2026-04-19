# NudeNet-based NSFW image detector
# Requires: pip install nudenet Pillow

import torch
from PIL import Image
from nudenet import NudeDetector

nude_detector = NudeDetector()

def detect_nsfw(image_path: str, nude_threshold: float = 0.35):
    """
    NudeNet NSFW detection.
    Returns: dict { "nudenet_score": float, "is_nsfw": bool }
    """
    pil_image = Image.open(image_path).convert("RGB")

    detect_result = nude_detector.detect(image_path)
    nude_score = max([r["score"] for r in detect_result], default=0.0)
    is_nsfw = (nude_score > nude_threshold)

    return {
        "nudenet_score": round(nude_score, 3),
        "is_nsfw": is_nsfw
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NudeNet NSFW detection")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()

    result = detect_nsfw(args.image, args.threshold)
    print(result)
    if result["is_nsfw"]:
        print("NSFW content detected.")
    else:
        print("Content is safe.")
