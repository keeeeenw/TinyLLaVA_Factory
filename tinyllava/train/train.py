from packaging import version
import pathlib

import tokenizers
import transformers


from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

import json
import os
from huggingface_hub import snapshot_download
def patch_rope_scaling_if_needed(model_name_or_path: str) -> str:
    """
    Ensures `config.json` uses the new Transformers RoPE schema:
    {"rope_scaling": {"type": "...", "factor": ...}}

    If `model_name_or_path` is a repo id, we download the snapshot to the HF cache
    and patch the cached config. If it's a local directory, we patch in place.

    Returns the local directory that contains the patched config.json.
    """
    # Resolve to a local directory with a config.json
    if os.path.isdir(model_name_or_path) and os.path.isfile(os.path.join(model_name_or_path, "config.json")):
        local_dir = model_name_or_path
    else:
        # Download only the config; reuse cache on subsequent runs
        # local_dir = snapshot_download(repo_id=model_name_or_path, allow_patterns=["config.json"])
        local_dir = snapshot_download(repo_id=model_name_or_path)

    config_path = os.path.join(local_dir, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    rope = config_dict.get("rope_scaling")
    if isinstance(rope, dict) and "rope_type" in rope and "type" not in rope:
        # Convert old keys to new schema
        new_rope = {
            "type": rope.get("rope_type", "dynamic"),
            "factor": float(rope.get("factor", 1.0)),
        }
        config_dict["rope_scaling"] = new_rope
        # Remove legacy field if present (not used by new schema)
        rope.pop("original_max_position_embeddings", None)

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"[rope_patch] Patched rope_scaling in {config_path}: {new_rope}")

    return local_dir


def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments) 
    return model_args

def _load_llm_settings(model_arguments):
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    llm_args['attn_implementation'] = model_arguments.attn_implementation # flash_attention_2 only supports torch.float16 and torch.bfloat16 dtypes
    return llm_args

def _load_vision_settings(model_arguments):
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args

def _load_connector_settings(model_arguments):
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    return connector_args


def train():
    
    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    
    logger_setting(getattr(training_arguments, 'output_dir', None))

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 
    # model_args contain arguements for huggingface model .from_pretrained function
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)

    # --- PATCH: normalize rope_scaling before TinyLlavaConfig loads the HF config ---
    # Update the path in model_arguments (and in model_args['llm'] if you pass it later)
    patched_llm_dir = patch_rope_scaling_if_needed(model_arguments.model_name_or_path)
    model_arguments.model_name_or_path = patched_llm_dir
    if "llm" in model_args:
        model_args["llm"]["model_name_or_path"] = patched_llm_dir
    # -------------------------------------------------------------------------------

    model_config = TinyLlavaConfig()
    model_config.load_from_config(model_arguments)
    model = TinyLlavaForConditionalGeneration(model_config)
    # load pretrained checkpoint
    if training_arguments.pretrained_model_path is not None:
        model = training_recipe.load(model, model_args)
    else:
        model.load_llm(**model_args['llm'])
        model.load_vision_tower(**model_args['vision_tower'])
        model.load_connector(**model_args['connector'])

    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model.tokenizer
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)
    log_trainable_params(model)  # not work well with zero3
    trainer = LLaVATrainer(model=model, #does not require model.to(device), huggingface/deepspeed does it for you?
                           tokenizer=tokenizer,
                           args=training_arguments,
                           **data_module)
    
    trainer.train()
    
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    train()
