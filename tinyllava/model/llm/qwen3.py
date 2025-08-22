from transformers import Qwen2ForCausalLM, AutoTokenizer

from . import register_llm

@register_llm('qwen3')
def return_qwen3class():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.unk_token = tokenizer.pad_token
        return tokenizer
    return Qwen2ForCausalLM, (AutoTokenizer, tokenizer_and_post_load)