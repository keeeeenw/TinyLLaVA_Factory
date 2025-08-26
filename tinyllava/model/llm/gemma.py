from transformers import GemmaForCausalLM, AutoTokenizer
from transformers import Gemma3ForCausalLM

from . import register_llm

@register_llm('gemma')
def return_gemmaclass():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer
    return (GemmaForCausalLM, (AutoTokenizer, tokenizer_and_post_load))

# Register for Gemma-3 models which use gemma3_text model type  
@register_llm('gemma-3')
def return_gemma3class():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer
    return (Gemma3ForCausalLM, (AutoTokenizer, tokenizer_and_post_load))
