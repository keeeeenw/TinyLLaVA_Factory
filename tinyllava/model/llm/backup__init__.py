import os
from transformers import AutoConfig

from ...utils import import_modules


LLM_FACTORY = {}


def LLMFactory(model_name_or_path):
    """
    Minimal tweaks:
      1) Stop at the first alias match
      2) Try basename as well as full string
      3) Fallback: read HF config and route by config.model_type
    """
    model, tokenizer_and_post_load = None, None
    lower_full = model_name_or_path.lower()
    lower_base = os.path.basename(lower_full)

    # First: match registered alias against full path or basename
    for name in LLM_FACTORY.keys():
        if name in lower_full or name in lower_base:
            model, tokenizer_and_post_load = LLM_FACTORY[name]()
            break

    # Fallback: infer backend by HF config.model_type (handles cache/local paths)
    if model is None:
        try:
            cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            mt = (getattr(cfg, "model_type", "") or "").lower()
            for name in LLM_FACTORY.keys():
                if name in mt:
                    model, tokenizer_and_post_load = LLM_FACTORY[name]()
                    break
        except Exception:
            pass

    assert model, f"{model_name_or_path} is not registered"
    return model, tokenizer_and_post_load


def register_llm(name):
    def register_llm_cls(cls):
        if name in LLM_FACTORY:
            return LLM_FACTORY[name]
        LLM_FACTORY[name] = cls
        return cls
    return register_llm_cls


# Automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.model.llm")

# Minimal alias so backend_key='llama' works even if a different alias was registered
for k in list(LLM_FACTORY.keys()):
    if "llama" in k and "llama" not in LLM_FACTORY:
        LLM_FACTORY["llama"] = LLM_FACTORY[k]
        break
