import argparse
import torch
import json
import os
import base64
from tqdm import tqdm
from datetime import datetime

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
import backoff

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


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def call_openai_api(*args, **kwargs):
    return client.chat.completions.create(*args, **kwargs)


def run_inference(
    item, image_file, args, model, tokenizer, image_processor, running_cost=0
):
    output_file = {}
    choices_str = ", ".join(
        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(item["choices"])]
    )
    if args.baseline is False:
        qs = f"""Solve this problem, and return the answer at the end of your response, e.g. Answer: A, B, C or D\n
              Problem: {DEFAULT_IMAGE_TOKEN}\n
              {item['problem_text']}\n
              Choices: {choices_str}"""
    else:
        if args.baseline_type == "direct":
            qs = f"""Please directly answer the question and provide the correct option letter, e.g., A, B, C, D.\n
                Question: {DEFAULT_IMAGE_TOKEN}\n
                {item['problem_text']}\n
                Choices: {choices_str}

                Answer:"""
        elif args.baseline_type == "cot":
            qs = f"""Please first conduct reasoning, and then answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
                Question: {DEFAULT_IMAGE_TOKEN}\n
                {item['problem_text']}\n
                Choices: {choices_str}

                Answer: Let's think step by step. """

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    prompt = conv.get_prompt()

    output_file["prompt"] = prompt
    output_file["image_file"] = image_file

    if args.model_path != "gpt-o":
        image = load_image(image_file)
        images_tensor = process_images([image], image_processor, model.config).to(
            model.device, dtype=torch.float16
        )
        output_file["image_tensor"] = images_tensor.shape
        raw_inputs, output_file = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
            output_file=output_file,
        )

        input_ids = raw_inputs.unsqueeze(0).cuda()
        output_file["input_ids"] = input_ids.shape

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            raw_output = model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=args.max_new_tokens,
                use_cache=False,
                stopping_criteria=[stopping_criteria],
                output_attentions=True,
                output_hidden_states=True,
                output_file=output_file,
            )

        output_file = raw_output["output_file"]
        output_ids = raw_output["main"].sequences
        attentions = raw_output["main"].attentions

        if args.single:
            torch.save(attentions, args.attention_weights_file)

            # output_text = ""
            # for generated_token_index, attention in enumerate(attentions):
            #     for i, decoder_element in enumerate(attention):
            #         output_text += f"Generated token index: {generated_token_index}, decoder element {i} shape: {decoder_element.shape}\n"

            # output_text += f"ATTENTION SHAPE: {len(attentions)}\n"
            # # Write the output to a text file
            # with open(f"attention_results/{timestamp}_attention.txt", "w") as file:
            #     file.write(output_text)

            # Initialize an empty list to store the output
            attention_description = []
            for generated_token_index, attention in enumerate(attentions):
                for i, decoder_element in enumerate(attention):
                    attention_description.append(
                        {
                            "generated_token_index": generated_token_index,
                            "decoder_element_index": i,
                            "decoder_element_shape": decoder_element.shape,
                        }
                    )

            # Convert output_ids to the words themselves
            output_words = [
                tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            output_file["output_tokens"] = output_words[0]
            output_file["all_tokens"] = (
                output_file["input_tokens"] + output_file["output_tokens"]
            )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()

            output_file["attention_description"] = attention_description
            output_file["outputs"] = outputs

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        output_file["attention_description"] = attention_description
        output_file["outputs"] = outputs

        # Write the output to a JSON file
        with open(args.attention_file, "w") as file:
            json.dump(output_file, file, indent=4)

    else:
        with open(image_file, "rb") as image_file:
            image_data = image_file.read()
            encoded_image = base64.b64encode(image_data).decode("utf-8")

        try:
            response = call_openai_api(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_image}"
                                },
                            },
                        ],
                    },
                ],
            )
            outputs = response.choices[0].message.content

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
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None, running_cost

    return outputs, running_cost


def eval_model(args):
    data_file = args.data_file

    # Load data file
    with open(data_file, "r") as f:
        data = json.load(f)

    results = []
    running_cost = 0

    # Initialize results file with an opening bracket
    with open(args.results_file, "w") as f:
        f.write("[\n")

    for index, item in enumerate(tqdm(data, desc="Processing items")):
        image_file = os.path.join(args.image_folder, item["image_id"] + ".png")
        outputs, running_cost = run_inference(
            item, image_file, args, model, tokenizer, image_processor, running_cost
        )
        extracted_answer = outputs
        if extracted_answer is None:
            continue

        # Answer Extraction
        choices_str = ", ".join(
            [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(item["choices"])]
        )
        prompt = f"""I will give you 4 choices, and a detailed answer. You must extract ONLY the letter (A, B, C or D) of the final answer from the detailed answer to the problem.
        
        Choices: {choices_str}

        Detailed Answer: {extracted_answer}

        Letter Answer:"""

        response = call_openai_api(
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

        print("Item: ", index)
        print("Detailed Answer: ", extracted_answer)
        print("Extracted Answer: ", extracted_answer)
        print("Running Cost: ", running_cost)
        print()  # Add a line break

        item["predicted_answer"] = extracted_answer
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
    parser.add_argument(
        "--baseline-type", type=str, choices=["direct", "cot"], required=False
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Specify if baseline is true or false",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Specify if the script should run inference independently for a single image file",
    )
    parser.add_argument("--idx", type=int, required=False)
    parser.add_argument("--attention-file", type=str, required=False)
    parser.add_argument("--attention-weights-file", type=str, required=False)
    args = parser.parse_args()

    if args.model_path != "gpt-o":
        # Model
        disable_torch_init()

        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, args.model_base, model_name
        )

    if args.single:
        print("SINGLE RUN!")
        data_file = args.data_file
        with open(data_file, "r") as f:
            data = json.load(f)

        item = data[args.idx]
        image_file = os.path.join(args.image_folder, item["image_id"] + ".png")

        outputs, running_cost = run_inference(
            item, image_file, args, model, tokenizer, image_processor
        )

        print("ATTENTION FILE: ", args.attention_file)
    else:
        eval_model(args)
