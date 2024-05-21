import argparse
import torch
import json
import os

from tinyllava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from tinyllava.conversation import conv_templates, SeparatorStyle
from tinyllava.model.builder import load_pretrained_model
from tinyllava.utils import disable_torch_init
from tinyllava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    data_file = args.data_file

    # Load data file
    with open(data_file, "r") as f:
        data = json.load(f)

    results = []
    running_cost = 0

    # Initialize results file with an opening bracket
    with open(args.results_file, "w") as f:
        f.write("[\n")

    for index, item in enumerate(data):
        if (
            item["problem_version"] == "Vision Only"
            or item["problem_version"] == "Text Lite"
        ):
            if item["question_type"] == "multi-choice":
                qs = f"""Solve this problem, and return the answer at the end of your response, e.g. Answer: A, B, C or D\n
                        Problem: {DEFAULT_IMAGE_TOKEN}\n
                        {item['question'] if 'question' in item and item['question'] else ''}\n"""
            elif item["question_type"] == "free-form":
                qs = f"""Solve this problem, and return the answer at the end of your response, e.g. Answer: 1, 2.5, 300.\n
                        Problem: {DEFAULT_IMAGE_TOKEN}\n
                        {item['question'] if 'question' in item and item['question'] else ''}\n"""

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image_file = os.path.join(args.image_folder, item["image"])
            image = load_image(image_file)
            images_tensor = process_images([image], image_processor, model.config).to(
                model.device, dtype=torch.float16
            )

            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()

            # Answer Extraction
            if item["question_type"] == "free-form":
                prompt = f"""I will give you a problem, and a detailed answer to that problem.

                You will extract the free-form answer (e.g. 2, 150.1, 300) from the detailed answer to the problem.
                
                Problem: {item['question']}
                Detailed Answer: {outputs}

                Free-Form Answer:"""
            elif item["question_type"] == "multi-choice":
                prompt = f"""I will give you a problem, and a detailed answer to that problem.

                You will extract the letter answer (A, B, C or D) from the detailed answer to the problem.
                
                Problem: {item['question']}
                Detailed Answer: {outputs}

                Letter Answer:"""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            # Extract token usage from the response
            total_tokens = response.usage.total_tokens
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            cost_per_million_input_tokens = 5  # $5 per 1 million input tokens
            cost_per_million_output_tokens = 15  # $15 per 1 million output tokens

            input_cost = (input_tokens / 1_000_000) * cost_per_million_input_tokens
            output_cost = (output_tokens / 1_000_000) * cost_per_million_output_tokens
            total_cost = input_cost + output_cost

            # Update running cost
            running_cost += total_cost

            # get answer
            extracted_answer = response.choices[0].message.content

            print("Item: ", item)
            print("Detailed Answer: ", outputs)
            print("Extracted Answer: ", extracted_answer)
            print("Running Cost: ", running_cost)
            print()  # Add a line break

            item["predicted_answer"] = outputs
            item["extracted_answer"] = extracted_answer
            results.append(item)

            # Append result to file
            with open(args.results_file, "a") as f:
                json.dump(item, f, indent=4)
                if index < len(data) - 1:
                    f.write(",\n")

    # Close the JSON array in the results file
    with open(args.results_file, "a") as f:
        f.write("\n]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--results-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=False)
    parser.add_argument("--conv-mode", type=str, default="phi")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)