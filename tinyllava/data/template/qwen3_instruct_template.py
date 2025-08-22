from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from . import register_template

from transformers import PreTrainedTokenizer
import torch
    
system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

@register_template('qwen3_instruct')
@dataclass
class Qwen3InstructTemplate(Template):
    """
    Qwen3 instruction template that uses USER/ASSISTANT format instead of ChatML.
    
    IMPORTANT NOTES:
    - This template uses USER/ASSISTANT format rather than the native Qwen3 ChatML format
      (<|im_start|>user/<|im_start|>assistant/<|im_end|>) due to tokenization mismatch issues
    - The separator must exactly match the format_user and format_assistant patterns for
      proper label masking (only assistant tokens are trained, user tokens masked with -100)
    - The separator[0] (' ASSISTANT: ') marks where assistant responses begin
    - The separator[1] ('<|im_end|>') is the end-of-turn token used to split conversation rounds
    - Changes to format_user/format_assistant MUST be accompanied by separator updates
    - If switching back to ChatML format, ensure the separator logic in base.py:_make_masks
      can properly parse the conversation structure to avoid "tokenization mismatch" warnings
    """
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT" + ": " + "{{content}}" + "<|im_end|>")
    system: "Formatter" = EmptyFormatter(slot=system+" ")
    separator: "Formatter" = EmptyFormatter(slot=[' ASSISTANT: ', '<|im_end|>'])