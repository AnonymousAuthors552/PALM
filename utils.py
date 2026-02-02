import json
import copy
from itertools import chain
from peft import PeftModel
from peft.tuners.lora.model import LoraModel
from dataclasses import fields, is_dataclass

# Copy from huggingface/datasets/src/datasets/utils/py_utils.py
# https://github.com/huggingface/datasets/blob/4.1.1/src/datasets/utils/py_utils.py
def asdict(obj):
    """Convert an object to its dictionary representation recursively.

    <Added version="2.4.0"/>
    """

    # Implementation based on https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict

    def _is_dataclass_instance(obj):
        # https://docs.python.org/3/library/dataclasses.html#dataclasses.is_dataclass
        return is_dataclass(obj) and not isinstance(obj, type)

    def _asdict_inner(obj):
        if _is_dataclass_instance(obj):
            result = {}
            for f in fields(obj):
                value = _asdict_inner(getattr(obj, f.name))
                if not f.init or value != f.default or f.metadata.get("include_in_asdict_even_if_is_default", False):
                    result[f.name] = value
            return result
        elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
            # obj is a namedtuple
            return type(obj)(*[_asdict_inner(v) for v in obj])
        elif isinstance(obj, (list, tuple)):
            # Assume we can create an object of this type by passing in a
            # generator (which is not true for namedtuples, handled
            # above).
            return type(obj)(_asdict_inner(v) for v in obj)
        elif isinstance(obj, dict):
            return {_asdict_inner(k): _asdict_inner(v) for k, v in obj.items()}
        else:
            return copy.deepcopy(obj)

    if not isinstance(obj, dict) and not _is_dataclass_instance(obj):
        raise TypeError(f"{obj} is not a dict or a dataclass")

    return _asdict_inner(obj)


def tokenize_function(tokenizer, example):
    return tokenizer(text=example["text"])

def concat(examples):    
    examples["input_ids"]=[list(chain.from_iterable(examples['input_ids']))] # convert chain to list of tokens
    examples["attention_mask"]=[list(chain.from_iterable(examples['attention_mask']))] # convert chain to list of tokens
    return examples

def chunk(examples):
    chunk_size = 1024 # modify this accordingly       
    input_ids = examples["input_ids"][0] # List[List], pass the inner list      
    attention_mask = examples["attention_mask"][0] # List[List]
    input_ids_truncated = []
    attention_mask_truncated = []
    
    #slice with step_size=chunk_size
    for i in range(0,len(input_ids),chunk_size):
        chunk = input_ids[i:i+chunk_size]
        if len(chunk)==chunk_size: # drop the last chunk if not equal
            input_ids_truncated.append(chunk)
            attention_mask_truncated.append(attention_mask[i:i+chunk_size])     
    examples['input_ids']=input_ids_truncated
    examples["attention_mask"]=attention_mask_truncated
        
    return examples   


def config_to_string(config_dict):
    for key, value in config_dict.items():
        if isinstance(value, set):
            config_dict[key] = list(value)
        elif isinstance(value, dict):
            config_dict[key] = {k: str(v) for k, v in value.items()}
        elif isinstance(value, list):
            config_dict[key] = [str(v) for v in value]
        else:
            try:
                json.dumps(value)
                config_dict[key] = value
            except TypeError:
                # Try to extract attributes directly, even for __slots__ classes like AddedToken
                if hasattr(value, "__getstate__"):
                    attr_dict = value.__getstate__()
                    try:
                        json.dumps(attr_dict)
                        config_dict[key] = attr_dict
                    except TypeError:
                        config_dict[key] = str(value)
                else:
                    config_dict[key] = str(value)
    return config_dict

def is_adapter_model(model) -> bool:
    """
    Check if the input model is a PEFT adapter model.

    Args:
        model: The model instance to check.

    Returns:
        True if the model is a PEFT adapter model, False otherwise.
    """
    
    return isinstance(model, (PeftModel, LoraModel))

def default_chat_template(conversation):
    prompt = ""
    
    for message in conversation:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            # Gemma format for system message
            prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        elif role == "user":
            prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        elif role == "assistant":
            prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
    
    # Add generation prompt for assistant response
    prompt += "<start_of_turn>model\n"
    
    return prompt

def are_tokenizers_identical(tokenizer1, tokenizer2, verbose=False):

    if hasattr(tokenizer1, "tokenizer"):
        tokenizer1 = tokenizer1.tokenizer
    if hasattr(tokenizer2, "tokenizer"):
        tokenizer2 = tokenizer2.tokenizer
    
    # If both are fast tokenizers, compare backend tokenizer JSON specs
    if getattr(tokenizer1, "is_fast", False) and getattr(tokenizer2, "is_fast", False):
        try:
            import json
            spec1 = json.loads(tokenizer1.backend_tokenizer.to_str())
            spec2 = json.loads(tokenizer2.backend_tokenizer.to_str())
            if spec1 != spec2:
                if verbose:
                    print("Fast tokenizer backend specs differ.")
                return False
            if verbose:
                print("Fast tokenizer backend specs match.")
            return True
        except Exception as e:
            if verbose:
                print(f"Error comparing fast tokenizers: {e}")
            return False

    # If either tokenizer is not fast, do manual comparison
    if verbose:
        print("The tokenizers are not a Fast tokenizer, falling back to manual comparison.")

    # 1. Compare vocabularies
    vocab1 = tokenizer1.get_vocab()
    vocab2 = tokenizer2.get_vocab()
    if vocab1 != vocab2:
        if verbose:
            print("Vocabularies differ.")
        return False
    if verbose:
        print("1. Vocabularies match.")

    # 2. Compare special tokens
    if tokenizer1.special_tokens_map != tokenizer2.special_tokens_map:
        if verbose:
            print("Special tokens map differs.")
        return False
    if verbose:
        print("2. Special tokens match.")

    # 3. Compare init configs (init_kwargs)
    config1 = tokenizer1.init_kwargs if hasattr(tokenizer1, "init_kwargs") else {}
    config2 = tokenizer2.init_kwargs if hasattr(tokenizer2, "init_kwargs") else {}
    if config1 != config2:
        if verbose:
            print("Init kwargs differ.")
        return False
    if verbose:
        print("3. Configs match.")

    return True