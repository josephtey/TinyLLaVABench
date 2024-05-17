import shortuuid
from datasets import load_dataset
from PIL import Image
import random
import json
import tqdm
import os

ds = load_dataset("AI4Math/MathVerse", "testmini")
data = []

image_path = "images/"
data_path = "mathverse_visual_only.json"

for sample_index, sample in enumerate(tqdm.tqdm(ds["testmini"])):
    if sample.get("problem_version") == "Vision Only":
        uuid = shortuuid.uuid()
        sample_dict = {
            "sample_index": sample_index,
            "problem_index": sample.get("problem_index", None),
            "problem_version": sample.get("problem_version", None),
            "question": sample.get("question", None),
            "answer": sample.get("answer", None),
            "question_type": sample.get("question_type", None),
            "image": uuid + ".png",  # Change extension to .png
        }
        # Convert image to RGB before saving as PNG
        sample_image = sample["image"].convert("RGB")
        sample_image.save(os.path.join(image_path, uuid + ".png"))
        data.append(sample_dict)

with open(data_path, "w") as f:
    json.dump(data, f, indent=4)
