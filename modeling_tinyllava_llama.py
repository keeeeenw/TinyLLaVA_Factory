"""
TinyLLaVA Standalone Model - Factory-Aligned Implementation
=========================================================

This file contains a standalone implementation of TinyLLaVA that replicates the behavior
of the factory-based model system without requiring the full factory infrastructure.

CRITICAL ALIGNMENT REQUIREMENTS:
===============================

To create a standalone model that produces identical results to the factory system,
the following components must be EXACTLY aligned:

1. PROMPT TEMPLATE FORMATTING:
   - Factory: "A chat between... USER: <image>\nQuestion ASSISTANT:" (NO SPACE after ASSISTANT:)
   - Bug: Adding space after "ASSISTANT: " causes repetitive, verbose generation
   - Fix: Use exact format "ASSISTANT:" without trailing space

2. TOKENIZATION:
   - Must use identical tokenizer_image_token() logic as factory
   - Handle BOS token offsets correctly
   - Use _insert_separator() function name (not insert_separator)

3. STOPPING CRITERIA:
   - Factory uses KeywordsStoppingCriteria with ["</s>"] keywords
   - Critical: Without stopping criteria, model generates repetitive loops
   - Must stop at EOS tokens and clean output by removing trailing "</s>"

4. IMAGE PROCESSING:
   - Process images as list: process_images([image], processor, config)
   - Handle both list and tensor outputs correctly
   - Apply proper device placement

5. GENERATION PARAMETERS:
   - Use identical parameters: temperature, top_p, num_beams, max_new_tokens
   - Same stopping criteria and output cleaning as factory

COMMON BUGS AND FIXES:
======================

BUG: Repetitive, numbered output (1. Be cautious... 2. Wet and muddy... 3. Noisy...)
FIX: Remove space after "ASSISTANT:" in prompt format

BUG: Model doesn't stop generating, creates very long responses
FIX: Add KeywordsStoppingCriteria with ["</s>"] keywords

BUG: Different results despite same architecture
FIX: Ensure exact prompt template matching factory system

BUG: Image not processed correctly
FIX: Pass images as list [image] not single image to process_images()

FACTORY SYSTEM COMPARISON:
=========================

Factory system uses:
- tinyllava.data.template.LlamaTemplate for prompt formatting
- tinyllava.utils.eval_utils.KeywordsStoppingCriteria for stopping
- tinyllava.eval.run_tiny_llava.eval_model() for inference pipeline

This standalone implementation replicates all these behaviors without dependencies.

USAGE:
======
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(path)
output, time = model.chat(prompt="Question?", image="path/url", tokenizer=tokenizer)
"""

import time

# Removed unused imports: dataclasses, Enum
from typing import List, Tuple, Optional, Union
import requests
from PIL import Image
from io import BytesIO
import base64
import re

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from transformers.utils import logging
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput, StoppingCriteria
from transformers import CLIPVisionModel, CLIPImageProcessor, SiglipVisionModel, SiglipImageProcessor

from .configuration import TinyLlavaConfig, IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM



logger = logging.get_logger(__name__)

# Model Constants (aligned with factory)
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

# Factory-aligned template formatting functions
def format_llama_prompt(question_list, answer_list, has_image=False):
    """Format prompt using factory template logic"""
    system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
    
    if isinstance(question_list, str):
        question_list = [question_list]
    if isinstance(answer_list, str):
        answer_list = [answer_list]
    
    msg = system
    for i, (question, answer) in enumerate(zip(question_list, answer_list)):
        # Format image token if present
        if DEFAULT_IMAGE_TOKEN in question:
            question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
            question = f"<image>\n{question}"
        
        # Format user and assistant messages
        msg += f"USER: {question} "
        if answer is not None:
            msg += f"ASSISTANT: {answer}</s>"
    
    return msg

def format_chat_prompt(prompt, has_image=False):
    """
    Format a single chat prompt for inference - matches factory template exactly.
    
    CRITICAL: This function replicates the exact prompt formatting used by:
    - tinyllava.data.template.LlamaTemplate
    - tinyllava.eval.run_tiny_llava.eval_model()
    
    CRITICAL BUG FIX: Must end with "ASSISTANT:" (NO SPACE)
    - Wrong: "ASSISTANT: " (with space) -> causes repetitive generation
    - Right: "ASSISTANT:" (no space) -> normal generation
    
    Args:
        prompt: User question/prompt
        has_image: Whether this prompt includes an image
    
    Returns:
        Formatted prompt string ready for tokenization
        
    Factory Template Equivalent:
        system + format_user.apply(content=formatted_prompt) + "ASSISTANT:"
        where format_user = "USER: {{content}} "
        and format_image_token = "<image>\n{{content}}"
    """
    # Exact system message from factory template (tinyllava/data/template/llama_template.py:17)
    system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
    
    if has_image:
        # Clean prompt and apply factory template format_image_token: "<image>\n{{content}}"
        clean_prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, '').strip() if DEFAULT_IMAGE_TOKEN in prompt else prompt.strip()
        formatted_prompt = f"<image>\n{clean_prompt}"
    else:
        formatted_prompt = prompt
    
    # Apply factory template format_user: "USER: {{content}} "
    # Then add ASSISTANT: for incomplete conversation (NO SPACE after ASSISTANT:)
    # CRITICAL: Space after ASSISTANT: causes generation issues!
    return system + f"USER: {formatted_prompt} ASSISTANT:"


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    """
    Tokenize prompt with image tokens, matching factory implementation exactly.
    
    CRITICAL: This function must match tinyllava.data.template.base.Template.tokenizer_image_token()
    
    Key details:
    - Function name must be _insert_separator (not insert_separator) to match factory
    - Handle BOS token offset correctly  
    - Process image tokens by replacing <image> with image_token_index
    
    Args:
        prompt: Text prompt with <image> tokens
        tokenizer: HuggingFace tokenizer
        image_token_index: Token ID for image placeholders (default: IMAGE_TOKEN_INDEX)
        return_tensors: Return format ('pt' for PyTorch tensor)
        
    Returns:
        List of token IDs or PyTorch tensor if return_tensors='pt'
        
    Factory equivalent: tinyllava.data.template.base.Template.tokenizer_image_token()
    """
    def _insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in _insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

ACT_TYPE = {
    'relu': nn.ReLU,
    'gelu': nn.GELU
}

class Connector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', config.connector_type)
        act_type = config.connector_type.split('_')[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.vision_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            
        self._connector = nn.Sequential(*modules)
    
    def forward(self, x):
        return self._connector(x)

class VisionTower(nn.Module):
    def __init__(self, cfg, model_name_or_path = 'clip'):
        super().__init__()
        if 'clip' in model_name_or_path:
            self._vision_tower = CLIPVisionModel(cfg)
            self._image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name_or_path)
        else:
            self._vision_tower = SiglipVisionModel(cfg)
            self._image_processor = SiglipImageProcessor.from_pretrained(cfg.model_name_or_path)
            
        self.config = cfg
        
    def forward(self, x, **kwargs):
        image_features = self._vision_tower(x, output_hidden_states=True)
        image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]

        if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
            image_features = image_features[:, 1:]
        elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")

        return image_features
        
    @property
    def vision_tower(self):
        return self._vision_tower
        
    @vision_tower.setter
    def vision_tower(self, vision_tower):
        self._vision_tower = vision_tower

def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None

class KeywordsStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria that stops generation when specific keywords are generated.
    
    CRITICAL: This class is essential for preventing repetitive generation.
    Without stopping criteria, the model will continue generating indefinitely,
    leading to repetitive, verbose output.
    
    Factory equivalent: tinyllava.utils.eval_utils.KeywordsStoppingCriteria
    
    The factory system uses this with keywords=["</s>"] to stop at EOS tokens.
    This prevents the model from generating beyond the natural response end.
    
    Args:
        keywords: List of stop words/tokens (typically ["</s>"])
        tokenizer: Tokenizer to encode keywords
        input_ids: Initial input tokens to track generation start
    """
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Check if any keyword appears at the end of generated sequence."""
        offset = min(input_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(input_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (input_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        return False
    

class TinyLlavaPreTrainedModel(PreTrainedModel):
    config_class = TinyLlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        return self.language_model._supports_sdpa


class TinyLlavaForConditionalGeneration(TinyLlavaPreTrainedModel):
    def __init__(self, config: TinyLlavaConfig):
        
        super().__init__(config)

        self.language_model = LlamaForCausalLM(config.text_config)
        self.vision_tower = VisionTower(config.vision_config, config.vision_model_name_or_path)
        self.connector = Connector(config)
        self.post_init()

    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        return self.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(inputs)

        return self.language_model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
    def encode_images(self, images):
        kwargs = {}
        kwargs['vision_feature_layer'] = self.config.vision_feature_layer
        kwargs['vision_feature_select_strategy'] = self.config.vision_feature_select_strategy
        images = images.to(device=self.device, dtype=self.dtype)
        image_features = self.vision_tower(images, **kwargs)
        image_features = self.connector(image_features)
        return image_features
    
    
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
        
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        
        image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.language_model.get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
    def chat(
        self,
        prompt: str,
        tokenizer = None,
        image: str = None,
        max_new_tokens: int = 512,
        num_beams = 1,
        top_p=None,
        temperature=0
    ):
        """
        Standalone chat interface that replicates factory system behavior exactly.
        
        CRITICAL FIXES APPLIED:
        =====================
        
        1. PROMPT FORMAT: Uses exact factory template format with "ASSISTANT:" (no space)
        2. STOPPING CRITERIA: Added KeywordsStoppingCriteria(["</s>"]) to prevent loops  
        3. IMAGE PROCESSING: Process images as [image] list, handle tensor outputs
        4. OUTPUT CLEANING: Strip EOS tokens like factory does
        
        This method replicates:
        - tinyllava.eval.run_tiny_llava.eval_model() pipeline
        - tinyllava.data.template.LlamaTemplate formatting
        - tinyllava.utils.eval_utils.KeywordsStoppingCriteria stopping
        
        Args:
            prompt: User question
            tokenizer: HuggingFace tokenizer  
            image: Image path/URL or None
            max_new_tokens: Maximum tokens to generate
            num_beams: Beam search width
            top_p: Nucleus sampling parameter
            temperature: Sampling temperature
            
        Returns:
            Tuple of (generated_text: str, generation_time: float)
            
        BUG HISTORY:
        ============
        - Original: Repetitive numbered lists due to wrong prompt format
        - Fixed: Exact factory template alignment prevents repetition
        """
        image_processor = self.vision_tower._image_processor

        # Format prompt using factory-aligned template
        has_image = image is not None
        # Don't add image token here - let format_chat_prompt handle it properly
        formatted_prompt = format_chat_prompt(prompt, has_image)
        
        image_tensor = None
        if image is not None:
            image = load_image(image)
            image_tensor = process_images([image], image_processor, self.config)
            if isinstance(image_tensor, list):
                image_tensor = torch.stack(image_tensor).to(self.device)
            else:
                image_tensor = image_tensor.to(self.device)

        # Tokenize using factory-aligned method
        input_ids = tokenizer_image_token(formatted_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        
        # Ensure proper shape and BOS token handling
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        
        # Generate
        stime = time.time()

        # Add stopping criteria to match factory behavior
        stop_str = "</s>"
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = self.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        generation_time = time.time() - stime
        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

        # Clean output like factory does
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs, generation_time
    

AutoConfig.register("tinyllava", TinyLlavaConfig)        
AutoModelForCausalLM.register(TinyLlavaConfig, TinyLlavaForConditionalGeneration)

"""
=============================================================================
STEP-BY-STEP GUIDE: Creating a Factory-Aligned Standalone Model
=============================================================================

To convert a factory-based TinyLLaVA model to a standalone HuggingFace model
that produces identical results, follow these steps:

STEP 1: Copy Factory Template Logic
===================================
- Copy prompt formatting from tinyllava/data/template/llama_template.py
- Key components:
  * system message (exact text with trailing space)
  * format_user = "USER: {{content}} "  
  * format_assistant = "ASSISTANT: {{content}}</s>"
  * format_image_token = "<image>\n{{content}}"

STEP 2: Fix Critical Prompt Format Bug  
======================================
CRITICAL: The prompt MUST end with "ASSISTANT:" (NO SPACE)
- Factory format: "...USER: <image>\nQuestion ASSISTANT:"
- Wrong format: "...USER: <image>\nQuestion ASSISTANT: " (causes repetition)
- This single space difference causes completely different generation behavior

STEP 3: Add Stopping Criteria
===============================
Copy KeywordsStoppingCriteria from tinyllava.utils.eval_utils
- Must stop at ["</s>"] tokens
- Without stopping criteria, model generates infinite repetitive loops
- Add to generate() call: stopping_criteria=[KeywordsStoppingCriteria(["</s>"], tokenizer, input_ids)]

STEP 4: Fix Tokenization
=========================  
Copy tokenizer_image_token from tinyllava.data.template.base
- Use _insert_separator (with underscore) function name
- Handle BOS token offsets correctly
- Process <image> tokens properly

STEP 5: Fix Image Processing
============================
- Pass images as list: process_images([image], processor, config)
- Handle both list and tensor return types
- Apply proper device placement: .to(self.device)

STEP 6: Add Output Cleaning
===========================
Clean outputs like factory does:
```python
outputs = outputs.strip()
if outputs.endswith(stop_str):
    outputs = outputs[:-len(stop_str)]
outputs = outputs.strip()
```

STEP 7: Test and Validate
=========================
Compare outputs between factory and standalone:
- Factory: python simply_inference.py
- Standalone: python hugging_face_inference.py
- Outputs should be nearly identical

DEBUGGING CHECKLIST:
====================
□ Prompt ends with "ASSISTANT:" (no space)
□ KeywordsStoppingCriteria added with ["</s>"]
□ Images processed as [image] list
□ _insert_separator function name used
□ Output cleaning implemented
□ Exact system message from factory template
□ Generation parameters match factory

RESULT COMPARISON:
==================
Before fixes: "1. Be cautious... 2. Wet and muddy... 3. Noisy... (repeats)"
After fixes:  "When I visit the beach at the waterfront, I should be cautious about several things. First, I should be cautious about the water..." (matches factory)

This documentation ensures future standalone models can be created without
repeating the debugging process that identified these critical alignment issues.
"""