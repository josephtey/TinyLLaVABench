from tinyllava.model.builder import load_pretrained_model
from tinyllava.mm_utils import get_model_name_from_path
from tinyllava.eval.run_tiny_llava import eval_model

# Paths
pretrained_model_path = (
    "/piech/u/joetey/TinyLLaVABench/finetuning/pre-trained/TinyLLaVA-3.1B"
)
lora_model_path = "/piech/u/joetey/TinyLLaVABench/finetuning/output/TinyLLaVA-3.1B-lora"
image_file = "images/2Ap9EEd8geGLDT6UQpjHGd.jpg,images/2aTMvJD8vdRaiFPYxh9Wte.jpg"  # Update with your image path
# image_file = "https://llava-vl.github.io/static/images/view.jpg"
prompt = "What is this?"

args = type(
    "Args",
    (),
    {
        "model_path": lora_model_path,
        "model_base": pretrained_model_path,
        "query": prompt,
        "conv_mode": "phi",
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
    },
)()

eval_model(args)
