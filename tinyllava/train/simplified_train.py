# Simplified training script for understanding the core components of the model training process.

import transformers
import torch
from tinyllava.train.llava_trainer import LLaVATrainer
from tinyllava.arguments import ModelArguments, DataArguments, TrainingArguments
from tinyllava.data.dataset import make_supervised_data_module
from tinyllava.model.model_factory import ModelSelect
from tinyllava.utils import rank0_print


def train():
    # Parse command line arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load the model
    if model_args.vision_tower:
        model = ModelSelect(model_args.model_name_or_path).from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.float16 if training_args.fp16 else torch.float32,
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path, cache_dir=training_args.cache_dir
        )

    # Disable caching to save memory
    model.config.use_cache = False

    # Prepare the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir
    )

    # Prepare data module
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Initialize the trainer
    trainer = LLaVATrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    # Start training
    trainer.train()

    # Save the trained model
    trainer.save_state()


if __name__ == "__main__":
    train()
