import base64
import os
import argparse
import json
import time
import io
import csv
import re
import psutil
import threading
import multiprocessing.util
import atexit
from tqdm import tqdm

from unsloth import FastLanguageModel, is_bfloat16_supported

import palm
from msh import *
import utils

import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from safetensors.torch import load_file

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import TrainingArguments, Trainer

from trl import SFTTrainer
from peft import PeftModel
import warnings

# Models use for the experiment
MODEL_MAP = {
    "llama": {
        "S": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "L": "unsloth/Meta-Llama-3.1-8B-Instruct",
    },
    "gemma": {
        "S": "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "L": "unsloth/gemma-3-4b-it",
    },
    "phi": {
        "S": "unsloth/Phi-4-mini-instruct-bnb-4bit",
        "L": "unsloth/Phi-4-mini-instruct",
    },
}

# Optional: models needing extra env tweaks
SPECIAL_ENV = {"gemma", "phi"}  

def get_model_path(args):
    if args.model in SPECIAL_ENV:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        pass  

    try:
        return MODEL_MAP[args.model][args.model_size]
    except KeyError:
        raise ValueError(
            f"No such model {args.model} size {args.model_size}. "
            f"Available = {list(MODEL_MAP.keys())}"
        )

def init_exp_config(args, 
                    model_path=None, 
                    mem_usage=0, 
                    total_access=0, 
                    compute_time=0, 
                    input_dataset_load_time=0, 
                    input_dataset_measure_time=0,
                    getitem_load_time=0,
                    getitem_measure_time=0,
                    input_model_load_time=0,
                    input_model_measure_time=0,
                    attestation_time=0,
                    output_dataset_storage_time=0,
                    output_dataset_measurement_time=0,
                    output_model_storage_time=0,
                    output_model_measurement_time=0,
                    output_storage_time=0,
                    output_measurement_time=0,
                    ):
    
    output_results = {
        "attestation_type": args.attestation_type,
        "model_path": model_path,
        "measure": args.measure,
        "in_memory": args.in_memory,
        "mem_usage": mem_usage,
        "total_access": total_access,
        "compute_time": compute_time,
        "input": {
            "input_dataset_load_time": input_dataset_load_time,
            "input_dataset_measure_time": input_dataset_measure_time,
            "getitem_load_time": getitem_load_time,
            "getitem_measure_time": getitem_measure_time,
            "input_model_load_time": input_model_load_time,
            "input_model_measure_time": input_model_measure_time,
        },
        "output": {
            "attestation_time": attestation_time,
            "output_dataset_storage_time": output_dataset_storage_time,
            "output_dataset_measurement_time": output_dataset_measurement_time,
            "output_model_storage_time": output_model_storage_time,
            "output_model_measurement_time": output_model_measurement_time,
            "output_total_storage_time": output_storage_time,
            "output_total_measure_time": output_measurement_time,
        },
    }
    return output_results

peak_memory_mb = 0
process = psutil.Process()

def monitor_memory(interval=0.1):
    global peak_memory_mb
    while True:
        mem = process.memory_info().rss / (1024 ** 3) 
        peak_memory_mb = max(peak_memory_mb, mem)
        time.sleep(interval)

monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
monitor_thread.start()

def start_memory_measure():
    # Measure process memory before
    in_process = psutil.Process(os.getpid())
    mem_before_proc = in_process.memory_info().rss / (1024 ** 3)

    # Measure system memory before
    mem_before_sys = psutil.virtual_memory().used / (1024 ** 3)

    print(f"[Process] Memory before: {mem_before_proc:.2f} GB")
    print(f"[System]  Total used RAM before: {mem_before_sys:.2f} GB")

    return in_process, mem_before_proc, mem_before_sys

def end_memory_measure(in_process, mem_before_proc, mem_before_sys):
    # Measure memory after
    mem_after_proc = in_process.memory_info().rss / (1024 ** 3)
    mem_after_sys = psutil.virtual_memory().used / (1024 ** 3)

    diff_proc = mem_after_proc - mem_before_proc

    print(f"[Process] Memory after: {mem_after_proc:.2f} GB")
    print(f"[Process] RAM memory used: {diff_proc:.2f} GB")

    print(f"[System]  Total used RAM after: {mem_after_sys:.2f} GB")
    print(f"[System]  Total RAM increased: {(mem_after_sys - mem_before_sys):.2f} GB")
    
    # atexit.register(lambda: report_peak_memory(mem_after_sys))
    # report_peak_memory(mem_after_sys)
    return diff_proc

class ProcessState:
    def __init__(self):
        self.running_access_hash = palm.MSetMuHash()
        # self.load_time = 0.0
        # self.measure_time = 0.0
        self.total_accesses = 0
        self.getitem_load_time = 0.0
        self.getitem_measure_time = 0.0
    def to_dict(self):
        return {
            "pid": os.getpid(),
            "total_accesses": self.total_accesses,
            # "load_time": self.load_time,
            # "measure_time": self.measure_time,
            "getitem_load_time": self.getitem_load_time,
            "getitem_measure_time": self.getitem_measure_time,
            "running_access_hash": str(self.running_access_hash.digest())
        }

def get_state():
    if not hasattr(get_state, "state"):
        get_state.state = multiprocessing.util.ForkAwareLocal()
    if not hasattr(get_state.state, "value"):
        get_state.state.value = ProcessState()
    return get_state.state.value

# Proof of Training experiment
def pretraining_attestation(args):
    
    path = "./data/bookcorpus"
    dataset_path = os.path.join(path, "dataset")
    tokenized_ds_path = os.path.join(path, "tokenized_ds")
    chunked_ds_path = os.path.join(path, "chunked_ds")

    model_path="gpt2"

    tokenizer_load_time_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer_load_time_end = time.time()
    tokenizer_load_time = tokenizer_load_time_end - tokenizer_load_time_start
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer_measure_time = 0

    if args.measure:
        # Measure the tokenizer
        tokenizer_measurement, tokenizer_measure_time = palm.save_tokenizer(tokenizer, "./", save_to_disk=False, measure=args.measure)

    while True:
        try:
            print("Loading with custom loader...")
            if not os.path.exists(chunked_ds_path):
                raise FileNotFoundError("Path not exists")

            # Measure the input dataset
            in_process, mem_before_proc, mem_before_sys = start_memory_measure()
            chunked_ds, chunked_ds_hashes = palm.load_dataset(load_path=chunked_ds_path, load_in_memory=args.in_memory, measure=args.measure)
            mem_usage = end_memory_measure(in_process, mem_before_proc, mem_before_sys)

            train_dataset = chunked_ds["train"]
            test_dataset = chunked_ds["test"]

            print("____________________________________")
            break

        except Exception as e:
            print(e)
            print("Data not found, downloading...")

            # loading raw data
            dataset = load_dataset("bookcorpus") #trust_remote_code=True
            dataset['train'] = dataset['train'].select(range(250000))
            dataset = dataset['train'].train_test_split(test_size=0.0015) 
            dataset.save_to_disk(dataset_path)
            
            tokenized_ds = dataset.map(lambda example: utils.tokenize_function(tokenizer, example),batched=True,remove_columns='text',num_proc=8)
            tokenized_ds.save_to_disk(tokenized_ds_path)
            concated_ds = tokenized_ds.map(utils.concat,batched=True,batch_size=1000000,num_proc=8)

            chunked_ds = concated_ds.map(utils.chunk,batched=True,batch_size=2,num_proc=8)
            chunked_ds.save_to_disk(chunked_ds_path)

            print("Download complete")

    data_collator = DataCollatorForLanguageModeling(tokenizer,mlm=False)

    compute_start = time.time()
    configuration = GPT2Config()
    model = GPT2LMHeadModel(configuration)

    training_args_dict = {
        "output_dir": "./saved_models/gpt-2-warm-up/standard-gpt",
        # "evaluation_strategy": "steps",
        "eval_steps": 500,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 8,
        "learning_rate": 2.5e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "weight_decay": 0.01,
        "logging_strategy": "steps",
        "logging_steps": 500,
        "save_steps": 5000,
        "save_total_limit": 10,
        "save_only_model": True,
        "remove_unused_columns": False,
        # "report_to": "wandb",
    }

    training_args = TrainingArguments(**training_args_dict) 
    
    print("Making Trainer() object...")
    trainer = Trainer(model=model,
                    args=training_args,
                    tokenizer=tokenizer,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    data_collator=data_collator)

    print("Training...")
    trainer.train()
    
    compute_end = time.time()
    
    compute_time = compute_end - compute_start
    print("Time to compute:", compute_time)

    attestation_time = 0
    output_measurement_time = 0
    output_storage_time = 0

    # Save and measure the output model
    directory = f"./saved_models/gpt2/"
    model_dir = os.path.join(directory, "model")
    evi_dir = os.path.join(directory, "evidence")
    if not os.path.exists(evi_dir):
        os.makedirs(evi_dir, exist_ok=True)
    
    output_model_time_start = time.time()
    model_hash, model_config_hash, output_model_measure_time = palm.save_model(trainer.model, model_dir, args.measure)
    palm.save_tokenizer(tokenizer, model_dir, measure=False)
    output_model_time_end = time.time()
    output_model_storage_time = (output_model_time_end - output_model_time_start) - output_model_measure_time
    print("Training Complete! Saved to:", directory)

    # Generate TDX TD quote using DCAP
    if args.measure:
        _, training_args_hash, training_args_measure_time = palm.measure_output(training_args.to_dict())
        print(f"----------\nTraining dataset final hash: {train_dataset.running_access_hash.digest()}")
        print(f"----------\nTesting dataset final hash: {test_dataset.running_access_hash.digest()}")
        print(f"Training dataset total accesses: {train_dataset.total_accesses}\n----------")
        print(f"Testing total accesses: {test_dataset.total_accesses}\n----------")

        print(f"Hash of model: {model_hash}")
        print(f"Hash of model's config: {model_config_hash}")
        
        print("--------------------------------------------")

        if train_dataset.in_memory and test_dataset.in_memory:
            chunked_ds_hashes_output = {
                split: {
                    fname: base64.b64encode(bytes.fromhex(h)).decode('utf-8')
                    for fname, h in chunked_ds_hashes[split].items()
                }
                for split in chunked_ds_hashes
            }
        else:
            h_train = train_dataset.running_access_hash.digest() 
            h_test = train_dataset.running_access_hash.digest() 
            h_train_bytes = h_train.to_bytes((h_train.bit_length() + 7) // 8, byteorder='big')
            h_test_bytes = h_test.to_bytes((h_test.bit_length() + 7) // 8, byteorder='big')
            chunked_ds_hashes_output = {
                'train': base64.b64encode(h_train_bytes).decode('utf-8'),
                'test': base64.b64encode(h_test_bytes).decode('utf-8')
            }
        payload = {
            'model_hash': base64.b64encode(model_hash).decode('utf-8'),
            'tokenizer_hash': {
                fname: base64.b64encode(h).decode('utf-8')
                for fname, (_, h, _) in tokenizer_measurement.items()
            },
            'dataset_hash': {
                'chunked_ds': chunked_ds_hashes_output
            },
            'training_configuration': {
                'model_config_hash': base64.b64encode(model_config_hash).decode('utf-8'),
                'training_args': base64.b64encode(training_args_hash).decode('utf-8'),
            }
        }
        with open(f"{evi_dir}/payload_pretrain.json", 'w') as f:
            json.dump(payload, f)
        output_measurement_time, attestation_time, output_storage_time = palm.attest(payload, f"{evi_dir}/attestation_{args.attestation_type}.json")

    # Experiment report
    exp_config = init_exp_config(args, 
                    model_path=model_path, 
                    mem_usage=mem_usage, 
                    total_access=(train_dataset.total_accesses + test_dataset.total_accesses), 
                    compute_time=compute_time, 
                    input_dataset_load_time=(train_dataset.load_time + test_dataset.load_time), 
                    input_dataset_measure_time=(train_dataset.measure_time + test_dataset.measure_time),
                    getitem_load_time=(train_dataset.getitem_load_time + test_dataset.getitem_load_time),
                    getitem_measure_time=(train_dataset.getitem_measure_time + test_dataset.getitem_measure_time),
                    input_model_load_time=tokenizer_load_time,
                    input_model_measure_time=(tokenizer_measure_time + training_args_measure_time),
                    attestation_time=attestation_time,
                    output_dataset_storage_time=0,
                    output_dataset_measurement_time=0,
                    output_model_storage_time=output_model_storage_time,
                    output_model_measurement_time=output_model_measure_time,
                    output_storage_time=(output_storage_time + output_model_storage_time),
                    output_measurement_time=(output_measurement_time + output_model_measure_time),
                    )

    return exp_config

# Proof of fine-tuning
def finetuning_attestation(args):

    path = "./data/yahma/alpaca-cleaned"
    dataset_path = os.path.join(path, "dataset")
    format_path = os.path.join(path, "formatted")

    model_path = get_model_path(args)
    print("Using model:", model_path)

    input_model_load_time_start = time.time()
    model, _ = FastLanguageModel.from_pretrained(model_name = model_path,
                                                 max_seq_length = 2048,
                                                 dtype = None,
                                                 load_in_4bit = False)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = FastLanguageModel.get_peft_model(model, 
                                             r = 16, 
                                             target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",], lora_alpha = 16, 
                                             lora_dropout = 0, 
                                             bias = "none", 
                                             use_gradient_checkpointing = "unsloth", 
                                             random_state = 3407, 
                                             use_rslora = False, 
                                             loftq_config = None)
    input_model_load_time_end = time.time()
    input_model_load_time = (input_model_load_time_end - input_model_load_time_start)

    print(f"Sucessfully loaded the model: {input_model_load_time}s")
    print("------------------------------------")

    if hasattr(model, "base_model"):
        if hasattr(model.base_model, "model"):
            base_model = model.base_model.model
        else:
            base_model = model.base_model
    else:
        base_model = model

    input_model_measure_time = 0

    if args.measure:
        # Measure the input model, LoRA, tokenizer
        og_model_hash, og_config_hash, og_model_measure_time = palm.save_model(model, "./", save_to_disk=False, measure=args.measure)
        og_base_model_hash, og_base_config_hash, og_base_model_measure_time = palm.save_model(base_model, "./", save_to_disk=False, measure=args.measure)
        tokenizer_measurement, tokenizer_measure_time = palm.save_tokenizer(tokenizer, "./", save_to_disk=False, measure=args.measure)

        input_model_measure_time += (og_model_measure_time + og_base_model_measure_time + tokenizer_measure_time)

        print(f"Hash of the original model: {og_model_hash}")
        print(f"Hash of the original model's config: {og_config_hash}")
        print("--------------------------------------------")


    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token 
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts}

    while True:
        try:
            print("Loading with custom loader...")
            if not os.path.exists(dataset_path) or not os.path.exists(format_path):
                raise FileNotFoundError(f"Dataset path '{dataset_path}' or format path '{format_path}' does not exist.")

            # Measure input dataset
            in_process, mem_before_proc, mem_before_sys = start_memory_measure()
            ds, formatted_ds_hashes = palm.load_dataset(load_path=format_path, load_in_memory=args.in_memory, measure=args.measure)
            mem_usage = end_memory_measure(in_process, mem_before_proc, mem_before_sys)

            print("____________________________________")
            break
        except Exception as e:
            print(e)
            print("Data not found, downloading...")
            # Loading raw data
            ds = load_dataset("yahma/alpaca-cleaned", split = "train")
            ds.save_to_disk(dataset_path)
            
            formatted_ds = ds.map(formatting_prompts_func, batched = True, num_proc=8)
            formatted_ds.save_to_disk(format_path)

    training_args_dict = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 50,
        "num_train_epochs": 1,
        # "max_steps": 20,
        "learning_rate": 2e-4,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "output_dir": "outputs",
        "report_to": "none",
        "remove_unused_columns": False,
    }
    training_args = TrainingArguments(**training_args_dict)

    compute_start = time.time()
    trainer = SFTTrainer(model = model,
                         tokenizer = tokenizer,
                         train_dataset = ds,
                         dataset_text_field = "text",
                         max_seq_length = 2048,
                         dataset_num_proc = 8,
                         packing = False,
                         args = training_args,
    )

    trainer.train()
    
    compute_end = time.time()
    compute_time = compute_end - compute_start
    print("Compute time:", compute_time)
    print("____________________________________")

    dataset = trainer.train_dataset
    
    attestation_time = 0
    output_measurement_time = 0
    output_storage_time = 0

    directory = os.path.join('./saved_models/', args.model)
    directory = os.path.join(directory, args.model_size)
    adapter_finetuned_model_path = os.path.join(directory, "finetuned_LoRA/")
    model_dir = os.path.join(adapter_finetuned_model_path, "model")
    evi_dir = os.path.join(adapter_finetuned_model_path, "evidence")
    if not os.path.exists(evi_dir):
        os.makedirs(evi_dir, exist_ok=True)
    
    # Measure output fine-tuned model
    output_model_storage_time_start = time.time()
    finetuned_model_hash, finetuned_config_hash, finetuned_model_measure_time = palm.save_model(trainer.model, model_dir, measure=args.measure)

    # Tokenizer is already measured during loading. No need to measure again
    palm.save_tokenizer(tokenizer, model_dir, measure=False)
    output_model_storage_time_end = time.time()
    output_model_storage_time = (output_model_storage_time_end - output_model_storage_time_start) - finetuned_model_measure_time

    print("Finetuning Complete! Saved to:", model_dir)

    # Generate TDX TD quote using DCAP
    if args.measure:
        _, training_args_hash, training_args_measure_time = palm.measure_output(training_args.to_dict())
        print("Total access:", dataset.total_accesses)
        print("Hash of the finetuned model:", finetuned_model_hash)
        print("Hash of the finetuned model's config:", finetuned_config_hash)
        print("--------------------------------------------")

        if dataset.in_memory:
            formatted_ds = {
                split: {
                    fname: base64.b64encode(bytes.fromhex(h)).decode('utf-8')
                    for fname, h in formatted_ds_hashes[split].items()
                }
                for split in formatted_ds_hashes
            }
        else:
            h = dataset.running_access_hash.digest() 
            h_bytes = h.to_bytes((h.bit_length() + 7) // 8, byteorder='big')
            formatted_ds = {
                'train': base64.b64encode(h_bytes).decode('utf-8')
            }
        payload = {
            'based_model': base64.b64encode(og_base_model_hash).decode('utf-8'),
            'original_model': base64.b64encode(og_model_hash).decode('utf-8'),
            'finetuned_model': base64.b64encode(finetuned_model_hash).decode('utf-8'),
            'tokenizer_hash': {
                fname: base64.b64encode(h).decode('utf-8')
                for fname, (_, h, _) in tokenizer_measurement.items()
            },
            'dataset_hash': {
                'formatted_ds': formatted_ds
            },
            'finetuning_configuration': {
                'base_model_config': base64.b64encode(og_base_config_hash).decode('utf-8'),
                'original_model_config': base64.b64encode(og_config_hash).decode('utf-8'),
                'finetuned_model_config': base64.b64encode(finetuned_config_hash).decode('utf-8'),
                'training_args': base64.b64encode(training_args_hash).decode('utf-8'),
            }
        }
        with open(f"{evi_dir}/payload_finetune.json", 'w') as f:
            json.dump(payload, f)
        output_measurement_time, attestation_time, output_storage_time = palm.attest(payload, f"{evi_dir}/attestation_{args.attestation_type}.json")

    exp_config = init_exp_config(args, 
                    model_path=model_path, 
                    mem_usage=mem_usage, 
                    total_access=dataset.total_accesses, 
                    compute_time=compute_time, 
                    input_dataset_load_time=dataset.load_time, 
                    input_dataset_measure_time=dataset.measure_time,
                    getitem_load_time=dataset.getitem_load_time,
                    getitem_measure_time=dataset.getitem_measure_time,
                    input_model_load_time=input_model_load_time,
                    input_model_measure_time=(input_model_measure_time + training_args_measure_time),
                    attestation_time=attestation_time,
                    output_dataset_storage_time=0,
                    output_dataset_measurement_time=0,
                    output_model_storage_time=output_model_storage_time,
                    output_model_measurement_time=finetuned_model_measure_time,
                    output_storage_time=(output_storage_time + output_model_storage_time),
                    output_measurement_time=(output_measurement_time + finetuned_model_measure_time),
                    )

    return exp_config

# Proof of inference
def inference_attestation(args):

    # Change the path to the dataset
    # ./data/coqa_1 for 1 sample (single inference)
    # ./data/coqa_50 for 10 samples with 5 questions each (session inference)
    dataset_path = "./data/coqa_1"
    model_path = get_model_path(args)
    print("Using model:", model_path)

    input_model_load_time_start = time.time()
    model, _ = FastLanguageModel.from_pretrained(model_name = model_path,
                                                 max_seq_length = 2048,
                                                 dtype = None,
                                                 load_in_4bit = False)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_model_load_time_end = time.time()
    input_model_load_time = input_model_load_time_end - input_model_load_time_start

    gen_kwargs = {
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
        # "early_stopping": True,
        "repetition_penalty": 1.05,
        "eos_token_id": None,
        "pad_token_id": None,
    }

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.sep_token or tokenizer.pad_token

        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

    directory = os.path.join('./saved_models/', args.model)
    adapter_og_model_path = os.path.join(directory, "original_LoRA/")
    input_model_measure_time = 0

    if args.measure:
        adapter_model_hash, adapter_config_hash, adapter_model_measure_time = palm.save_model(model, adapter_og_model_path, save_to_disk=False, measure=args.measure)
        tokenizer_measurement, tokenizer_measure_time = palm.save_tokenizer(tokenizer, adapter_og_model_path, save_to_disk=False, measure=args.measure)

        _, inference_config_hash, inference_config_measure_time = palm.measure_output(gen_kwargs)

        input_model_measure_time += (adapter_model_measure_time + tokenizer_measure_time + inference_config_measure_time)

    while True:
        try:
            print("Loading with custom loader...")
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")

            in_process, mem_before_proc, mem_before_sys = start_memory_measure()
            dataset, dataset_hashes = palm.load_dataset(load_path=dataset_path, load_in_memory=args.in_memory, measure=args.measure)
            mem_usage = end_memory_measure(in_process, mem_before_proc, mem_before_sys)

            print("____________________________________")
            break
        except Exception as e:
            print(e)
            print("Data not found, downloading...")
            # Loading raw data
            dataset = load_dataset("stanfordnlp/coqa", split="validation")
            dataset.save_to_disk(dataset_path)
            # subset = dataset.select(range(10))
            # subset.save_to_disk(dataset_path)
            
    FastLanguageModel.for_inference(model)

    all_conversations = [
        {"role": "system", "content": "You are a helpful assistant. Your task is to answer the question."},
    ]

    def generate_text(conversation):
        # Apply the chat template to format the prompt
        try:
            prompt = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            print(f"Chat template failed ({e}), using custom template.")
            prompt = palm.default_chat_template(conversation)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_kwargs
            )
        
        generated_sequence = outputs[0]
        input_len = inputs['input_ids'].shape[1]
        new_tokens = generated_sequence[input_len:]

        reply = tokenizer.decode(new_tokens, skip_special_tokens=False)

        # Clean up all common stop tokens
        stop_tokens = [
            "<|eot_id|>", "<|end_of_text|>", "</s>", "<|im_end|>", "<|end|>", 
            "<end_of_turn>", "<start_of_turn>", "<|start_header_id|>"
        ]

        for stop in stop_tokens:
            index = reply.find(stop)
            if index != -1:
                reply = reply[:index]
                break 

        return reply

    compute_time = 0

    # user_input_hashes = []
    
    for sample in dataset:
        conversation = [
            {"role": "system", "content": f"You are a helpful assistant. Your task is to answer the question with this context: {sample["story"]}"},
        ]
        # [:1] for single and [:5] for session 
        for prompt in sample["questions"][:5]: #[:1] 
            conversation.append({"role": "user", "content": prompt})
            compute_time_start = time.time()
            reply = generate_text(conversation)
            compute_time_end = time.time()
            compute_time += (compute_time_end - compute_time_start)

            print(f"User: {prompt}")
            print(f"Assistant: {reply}\n")

            conversation.append({"role": "assistant", "content": reply})
        all_conversations.append(conversation)
        print("----------------------------------------------\n")
    
    # # This code is for direct user prompt as an input
    # while True:
    #     prompt = input("User: ")
    #     if prompt.lower() in ["exit", "quit"]:
    #         break

    #     conversation.append({"role": "user", "content": prompt})
    #     compute_time_start = time.time()
    #     reply = generate_text(conversation)
    #     compute_time_end = time.time()
    #     compute_time += (compute_time_end - compute_time_start)
    #     print(f"Assistant: {reply}")

    #     conversation.append({"role": "assistant", "content": reply})
    # print(all_conversations)

    directory = f"./llm_output/{args.attestation_type}"
    directory = os.path.join(directory, f"{args.model}_{args.model_size}_{args.in_memory}")
    evi_dir = os.path.join(directory, "evidence")
    if not os.path.exists(evi_dir):
        os.makedirs(evi_dir, exist_ok=True)

    # Measure the conversation output
    conv_output_storage_time_start = time.time()
    with open(f"{directory}/conversation.json", 'w') as f:
        json.dump(all_conversations, f, indent=2, default=str)
    conv_output_storage_time_end = time.time()
    conv_output_storage_time = conv_output_storage_time_end - conv_output_storage_time_start
    print("Saved conversation")

    attestation_time = 0
    output_measurement_time = 0
    output_storage_time = 0

    # Generate TDX TD quote using DCAP
    if args.measure:
        _, conv_hash, conv_measure_time = palm.measure_output(all_conversations)
        payload = {
            'model': base64.b64encode(adapter_model_hash).decode('utf-8'),
            'tokenizer_hash': {
                fname: base64.b64encode(h).decode('utf-8')
                for fname, (_, h, _) in tokenizer_measurement.items()
            },
            'conversation': base64.b64encode(conv_hash).decode('utf-8'),
            'configuration': {
                'model_config': base64.b64encode(adapter_config_hash).decode('utf-8'),
                'inference_config': base64.b64encode(inference_config_hash).decode('utf-8')
            }
        }
        with open(f'{evi_dir}/payload_inference.json', 'w') as f:
            json.dump(payload, f)
        output_measurement_time, attestation_time, output_storage_time = palm.attest(payload, f"{evi_dir}/attestation_{args.attestation_type}.json")

    exp_config = init_exp_config(args, 
                    model_path=model_path, 
                    mem_usage=mem_usage, 
                    total_access=dataset.total_accesses, 
                    compute_time=compute_time, 
                    input_dataset_load_time=dataset.load_time, 
                    input_dataset_measure_time=dataset.measure_time,
                    getitem_load_time=dataset.getitem_load_time,
                    getitem_measure_time=dataset.getitem_measure_time,
                    input_model_load_time=input_model_load_time,
                    input_model_measure_time=input_model_measure_time,
                    attestation_time=attestation_time,
                    output_dataset_storage_time=0,
                    output_dataset_measurement_time=0,
                    output_model_storage_time=0,
                    output_model_measurement_time=0,
                    output_storage_time=(output_storage_time + conv_output_storage_time),
                    output_measurement_time=(output_measurement_time + conv_measure_time),
                    )

    return exp_config

# Proof of evaluation (MMLU)
def evaluation_attestation(args):
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager, get_task_dict, ConfigurableTask, ConfigurableGroup
    from lm_eval.models.huggingface import HFLM
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    model_path = get_model_path(args)
    print("Using model:", model_path)

    input_model_load_time_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_model_load_time_end = time.time()
    input_model_load_time = input_model_load_time_end - input_model_load_time_start
    input_model_measure_time = 0

    # Measure the input model and tokenizer
    if args.measure:
        model_hash, model_config_hash, model_measure_time = palm.save_model(model, "./", save_to_disk=False, measure=args.measure)
        tokenizer_measurement, tokenizer_measure_time = palm.save_tokenizer(tokenizer, "./", save_to_disk=False, measure=args.measure)
        input_model_measure_time += (model_measure_time + tokenizer_measure_time)


    model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        # peft=model,
        batch_size=1,
        max_length=4096,
    )

    eval_tasks = [
        "mmlu",
        # "arc_easy",
        # "arc_challenge",
        # "hellaswag",
        # "truthfulqa_gen",
        # "openbookqa",
        # "winogrande",
        # "piqa",
        # "gsm8k",
        # "squadv2",
        # "lambada_openai",
    ]

    task_manager = TaskManager()
    task_dict = {}
    dataset_hashes = {}

    total_records = 0
    
    def flatten_task_or_group(obj):
        """
        Recursively flatten ConfigurableGroup into individual ConfigurableTask(s) (if there is more than one task).
        Returns a dict of task_name -> ConfigurableTask.
        """
        flat_tasks = {}

        if isinstance(obj, ConfigurableTask):
            flat_tasks[obj.task_name] = obj

        elif isinstance(obj, ConfigurableGroup):
            for sub_obj in obj.values():
                flat_tasks.update(flatten_task_or_group(sub_obj))

        elif isinstance(obj, dict):
            for sub_obj in obj.values(): 
                flat_tasks.update(flatten_task_or_group(sub_obj))

        else:
            raise ValueError(f"Unexpected object type: {type(obj)}")

        return flat_tasks
    
    def get_task(task_name: str, task_manager: TaskManager):
        nonlocal total_records
        # task_or_group = get_task_dict([task_name], task_manager=task_manager)[task_name]
        task_dict_manager = get_task_dict([task_name], task_manager=task_manager)
        tasks_obj = next(iter(task_dict_manager.values()))

        # Flatten groups into individual tasks
        task_map = flatten_task_or_group(tasks_obj)

        for t_name, task in tqdm(task_map.items(), desc=f"Preparing Dataset"):
            ds = {}
            for split in task.dataset:
                if args.in_memory:
                    input_dataset_load_time_start = time.time()
                    d = Dataset.from_list(task.dataset[split])
                    input_dataset_load_time_end = time.time()
                    d_intput_load_time = input_dataset_load_time_end - input_dataset_load_time_start

                    total_records += len(d)

                    _, dataset_hash, ds_input_measure_time = palm.measure_output(d.to_list())
                    dataset_hashes[f"{t_name}:{split}"] = dataset_hash

                    ds[split] = palm.MeasureDataset(
                        d,
                        load_in_memory=args.in_memory,
                        data_load_time=d_intput_load_time,
                        data_measure_time=ds_input_measure_time,
                        measure=args.measure,
                    )
                else:
                    ds[split] = palm.MeasureDataset(task.dataset[split], load_in_memory=args.in_memory, measure=args.measure)

            task.dataset = ds

        return task_map

    in_process, mem_before_proc, mem_before_sys = start_memory_measure()
    task_dict = {}
    for task_name in eval_tasks:
        print(task_name)
        tasks = get_task(task_name, task_manager)
        task_dict.update(tasks) 

    mem_usage = end_memory_measure(in_process, mem_before_proc, mem_before_sys)

    print(task_dict)
    print(total_records)

    compute_time_start = time.time()
    results = evaluator.evaluate(
        lm=model,
        task_dict=task_dict,
    )
    compute_time_end = time.time()
    compute_time = compute_time_end - compute_time_start
    print("Compute time:", compute_time)

    input_dataset_load_time = 0
    input_dataset_measure_time = 0
    getitem_load_time = 0
    getitem_measure_time = 0
    total_accesses = 0
    for task_name, task in task_dict.items(): 
        print(task)
        for split, dataset in task.dataset.items():
            input_dataset_load_time += dataset.load_time
            input_dataset_measure_time += dataset.measure_time
            getitem_load_time += dataset.getitem_load_time
            getitem_measure_time += dataset.getitem_measure_time
            total_accesses += dataset.total_accesses

    print(lm_eval.utils.make_table(results))

    directory = f"./llm_output/{args.attestation_type}"
    directory = os.path.join(directory, f"{args.model}_{args.model_size}")
    evi_dir = os.path.join(directory, "evidence")
    if not os.path.exists(evi_dir):
        os.makedirs(evi_dir, exist_ok=True)

    attestation_time = 0
    output_measurement_time = 0
    output_storage_time = 0

    # Measure the output MMLU result
    if args.measure:
        _, eval_result_hash, result_output_measure_time = palm.measure_output(results)

    result_output_time_start = time.time()
    with open(f'{evi_dir}/lm_eval_results_{args.model}_{args.model_size}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    result_output_time_end = time.time()
    result_output_storage_time = result_output_time_end - result_output_time_start
    print("Saved")

    # Generate TDX TD quote using DCAP
    if args.measure:
        final_hashes = {}

        for task_name, task in task_dict.items(): 
            for split, dataset in task.dataset.items():
                key = f"{task_name}:{split}" 

                if dataset.in_memory:
                    hash_bytes = dataset_hashes[key]
                    final_hashes[key] = base64.b64encode(hash_bytes).decode('utf-8')
                else:
                    h = dataset.running_access_hash.digest()
                    h_bytes = h.to_bytes((h.bit_length() + 7) // 8, byteorder='big')
                    final_hashes[key] = base64.b64encode(h_bytes).decode('utf-8')
        payload = {
            'model': base64.b64encode(model_hash).decode('utf-8'),
            'tokenizer_hash': {
                fname: base64.b64encode(h).decode('utf-8')
                for fname, (_, h, _) in tokenizer_measurement.items()
            },
            'dataset_hashes': final_hashes,
            'configuration': {
                'model_config': base64.b64encode(model_config_hash).decode('utf-8'),
            },
            'evaluation': {
                'results_hash': base64.b64encode(eval_result_hash).decode('utf-8')
            }
        }
        with open('payload_evaluation.json', 'w') as f:
            json.dump(payload, f)
        output_measurement_time, attestation_time, output_storage_time = palm.attest(payload, f"{evi_dir}/attestation_{args.attestation_type}.json")

    exp_config = init_exp_config(args, 
                    model_path=model_path, 
                    mem_usage=mem_usage, 
                    total_access=total_accesses, 
                    compute_time=compute_time, 
                    input_dataset_load_time=input_dataset_load_time, 
                    input_dataset_measure_time=input_dataset_measure_time,
                    getitem_load_time=getitem_load_time,
                    getitem_measure_time=getitem_measure_time,
                    input_model_load_time=input_model_load_time,
                    input_model_measure_time=input_model_measure_time,
                    attestation_time=attestation_time,
                    output_dataset_storage_time=0,
                    output_dataset_measurement_time=0,
                    output_model_storage_time=0,
                    output_model_measurement_time=0,
                    output_storage_time=(output_storage_time + result_output_storage_time),
                    output_measurement_time=(output_measurement_time + result_output_measure_time),
                    )

    return exp_config

# Proof of evaluation (BLEU)
def eval_bleu(args):
    import evaluate

    model_path = get_model_path(args)
    print("Using model:", model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = "wmt/wmt14"
    lang_pair = "de-en"
    path = os.path.join("./data/", dataset_name)
    dataset_path = os.path.join(path, lang_pair)

    input_model_load_time_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_model_load_time_end = time.time()
    input_model_load_time = input_model_load_time_end - input_model_load_time_start

    input_model_measure_time = 0

    # Measure the input model and tokenizer
    if args.measure:
        model_hash, config_hash, model_measure_time = palm.save_model(model, "./", save_to_disk=False, measure=args.measure)
        tokenizer_measurement, tokenizer_measure_time = palm.save_tokenizer(tokenizer, "./", save_to_disk=False, measure=args.measure)
        input_model_measure_time += (model_measure_time + tokenizer_measure_time)

    model = model.to(device)
    
    while True:
        try:
            print("Loading dataset...")
            # Measure the input dataset
            in_process, mem_before_proc, mem_before_sys = start_memory_measure()
            dataset, dataset_hashes = palm.load_dataset(load_path=dataset_path, load_in_memory=args.in_memory, measure=args.measure)
            mem_usage = end_memory_measure(in_process, mem_before_proc, mem_before_sys)
            break
        except Exception as e:
            print(f"Fallback to HF download: {e}")
            dataset = load_dataset(dataset_name, lang_pair, split="test")
            dataset.save_to_disk(dataset_path)
            dataset.cleanup_cache_files()
            print("Saved dataset to disk.")

    metric = evaluate.load("sacrebleu")
    
    predictions = []
    references = []
    max_new_tokens = 80
    src_lang, ref_lang = lang_pair.split("-")

    few_shot_examples = [
        ("Ich liebe Programmierung.", "I love programming."),
        ("Guten Morgen!", "Good morning!"),
    ]

    compute_start = time.time()
    for record in tqdm(dataset, desc="Evaluating BLEU"):
        src = record["translation"][src_lang]
        ref = record["translation"][ref_lang]

        # Build few-shot prompt
        prompt = f"Translate the following {src_lang} sentences into {ref_lang}.\n\n"
        for ex_src, ex_tgt in few_shot_examples:
            prompt += f"{src_lang}: {ex_src}\n{ref_lang}: {ex_tgt}\n\n"
        prompt += f"{src_lang}: {src}\n{ref_lang}:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        pred_full = tokenizer.decode(out[0], skip_special_tokens=True)

        prompt_suffix = f"{src_lang}: {src}\n{ref_lang}:"
        pred_after_input = pred_full.split(prompt_suffix)[-1].strip()

        pred_lines = pred_after_input.split("\n")
        pred = pred_lines[0].strip()

        predictions.append(pred)
        references.append([ref])

    compute_end = time.time()
    compute_time = compute_end - compute_start
    print("Compute time:", compute_time)


    references = [[r] for r in references]
    result = metric.compute(predictions=predictions, references=references)
    print("BLEU score:", result["score"])
    print(result)

    directory = f"./llm_output/{args.attestation_type}"
    directory = os.path.join(directory, f"{args.model}_{args.model_size}")
    evi_dir = os.path.join(directory, "evidence")
    if not os.path.exists(evi_dir):
        os.makedirs(evi_dir, exist_ok=True)

    # Measure output BLEU score result
    if args.measure:
        _, eval_result_hash, eval_result_output_measure_time = palm.measure_output(result)

    eval_result_output_storage_time_start = time.time()
    with open(f'{directory}/eval_bleu_results_{args.model}_{args.model_size}.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    eval_result_output_storage_time_end = time.time()
    eval_result_output_storage_time = eval_result_output_storage_time_end - eval_result_output_storage_time_start
    print("Saved BLEU results to:", directory)

    attestation_time = 0
    output_measurement_time = 0
    output_storage_time = 0

    # Generate TDX TD quote using DCAP
    if args.measure:
        if dataset.in_memory:
            ds_hashes = {
                split: {
                    fname: base64.b64encode(bytes.fromhex(h)).decode('utf-8')
                    for fname, h in dataset_hashes[split].items()
                }
                for split in dataset_hashes
            }
        else:
            h = dataset.running_access_hash.digest() 
            h_bytes = h.to_bytes((h.bit_length() + 7) // 8, byteorder='big')
            ds_hashes = {
                'train': base64.b64encode(h_bytes).decode('utf-8')
            }
        payload = {
            'model': base64.b64encode(model_hash).decode('utf-8'),
            'tokenizer_hash': {
                fname: base64.b64encode(h).decode('utf-8')
                for fname, (_, h, _) in tokenizer_measurement.items()
            },
            'dataset_hashes': ds_hashes,
            'configuration': {
                'model_config': base64.b64encode(config_hash).decode('utf-8'),
            },
            'evaluation': {
                'results_hash': base64.b64encode(eval_result_hash).decode('utf-8')
            }
        }
        with open(f'{evi_dir}/payload_evaluation.json', 'w') as f:
            json.dump(payload, f)
        output_measurement_time, attestation_time, output_storage_time = palm.attest(payload, f"{evi_dir}/attestation_{args.attestation_type}.json")

    exp_config = init_exp_config(args, 
                    model_path=model_path, 
                    mem_usage=mem_usage, 
                    total_access=dataset.total_accesses, 
                    compute_time=compute_time, 
                    input_dataset_load_time=dataset.load_time, 
                    input_dataset_measure_time=dataset.measure_time,
                    getitem_load_time=dataset.getitem_load_time,
                    getitem_measure_time=dataset.getitem_measure_time,
                    input_model_load_time=input_model_load_time,
                    input_model_measure_time=input_model_measure_time,
                    attestation_time=attestation_time,
                    output_dataset_storage_time=0,
                    output_dataset_measurement_time=0,
                    output_model_storage_time=0,
                    output_model_measurement_time=0,
                    output_storage_time=(output_storage_time + eval_result_output_storage_time),
                    output_measurement_time=(output_measurement_time + eval_result_output_measure_time),
                    )

    return exp_config

# Proof of quantization
def quantization_attestation(args):
    from awq import AutoAWQForCausalLM

    model_path = get_model_path(args)
    print("Using model:", model_path)

    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

    input_model_load_time_start = time.time()
    awq_model = AutoAWQForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_model_load_time_end = time.time()
    input_model_load_time = input_model_load_time_end - input_model_load_time_start

    input_model_measure_time = 0

    # Measure the input model, tokenizer, and quanzation configuration
    if args.measure:
        og_model_hash, og_config_hash, og_model_measure_time = palm.save_model(awq_model.model, "./", save_to_disk=False, measure=args.measure)
        tokenizer_measurement, tokenizer_measure_time = palm.save_tokenizer(tokenizer, "./", save_to_disk=False, measure=args.measure)
        _, quantize_config_hash, quantize_config_measure_time = palm.measure_output(quant_config)

        input_model_measure_time = og_model_measure_time + tokenizer_measure_time + quantize_config_measure_time

    compute_start = time.time()
    awq_model.quantize(tokenizer, quant_config=quant_config)
    compute_end = time.time()
    compute_time = compute_end - compute_start
    print("Compute time:", compute_time)
    print("____________________________________")

    directory = os.path.join('./saved_models/', args.model)
    directory = os.path.join(directory, args.model_size)
    model_dir = os.path.join(directory, "quantized_awq/")
    evi_dir = os.path.join(model_dir, "evidence")
    if not os.path.exists(evi_dir):
        os.makedirs(evi_dir, exist_ok=True)

    # Save and measure the quantized model
    output_model_storage_time_start = time.time()
    quantized_model_hash, quantized_model_config_hash, quantized_model_measure_time = palm.save_model(awq_model.model, model_dir, measure=args.measure)
    output_model_storage_time_end = time.time()
    output_model_storage_time = (output_model_storage_time_end - output_model_storage_time_start) - quantized_model_measure_time

    print("Finetuning Complete! Saved to:", model_dir)
    print("--------------------------------------------")

    attestation_time = 0
    output_measurement_time = 0
    output_storage_time = 0

    # Generate TDX TD quote using DCAP
    if args.measure:
        print(f"Hash of the quantized model: {quantized_model_hash}")
        print(f"Hash of the quantized model's config: {quantized_model_config_hash}")
        payload = {
            'original_model': base64.b64encode(og_model_hash).decode('utf-8'),
            'quantized_model': base64.b64encode(quantized_model_hash).decode('utf-8'),
            'tokenizer_hash': {
                fname: base64.b64encode(h).decode('utf-8')
                for fname, (_, h, _) in tokenizer_measurement.items()
            },
            'finetuning_configuration': {
                'original_model_config': base64.b64encode(og_config_hash).decode('utf-8'),
                'quantized_model_config': base64.b64encode(quantized_model_config_hash).decode('utf-8'),
                'quantization_config': base64.b64encode(quantize_config_hash).decode('utf-8')
            }
        }
        with open(f'{evi_dir}/payload_quantization.json', 'w') as f:
            json.dump(payload, f)
        output_measurement_time, attestation_time, output_storage_time = palm.attest(payload, f"{evi_dir}/attestation_{args.attestation_type}.json")

    exp_config = init_exp_config(args, 
                    model_path=model_path, 
                    mem_usage=0, 
                    total_access=0, 
                    compute_time=compute_time, 
                    input_dataset_load_time=0, 
                    input_dataset_measure_time=0,
                    getitem_load_time=0,
                    getitem_measure_time=0,
                    input_model_load_time=input_model_load_time,
                    input_model_measure_time=input_model_measure_time,
                    attestation_time=attestation_time,
                    output_dataset_storage_time=0,
                    output_dataset_measurement_time=0,
                    output_model_storage_time=output_model_storage_time,
                    output_model_measurement_time=quantized_model_measure_time,
                    output_storage_time=(output_storage_time + output_model_storage_time),
                    output_measurement_time=(output_measurement_time + quantized_model_measure_time),
                    )

    return exp_config

# Proof of dataset attribute distribution
def distribution_attestation(args):
    from collections import Counter
    import nltk
    import matplotlib.ticker as mticker
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words('english'))
    word_pattern = re.compile(r'\b[a-zA-Z]+\b')

    dataset_name = "bookcorpus"
    path = os.path.join("./data/", dataset_name)
    dataset_path = os.path.join(path, "dataset")

    while True:
        try:
            print("Loading dataset...")
            print(f"In-memory: {args.in_memory}")
            in_process, mem_before_proc, mem_before_sys = start_memory_measure()
            dataset, dataset_hashes = palm.load_dataset(load_path=dataset_path, load_in_memory=args.in_memory, split="train", measure=args.measure)
            mem_usage = end_memory_measure(in_process, mem_before_proc, mem_before_sys)
            break
        except Exception as e:
            print(f"Fallback to HF download: {e}")
            dataset = load_dataset(dataset_name, split="train")
            dataset.save_to_disk(dataset_path)
            dataset.cleanup_cache_files()
            print("Saved dataset to disk.")

    print(f"Dataset load time: {dataset.load_time}")
    print(f"Dataset measure time: {dataset.measure_time}")
    print(f"Total accesses: {dataset.total_accesses}")
    print(f"Access Trace: {dataset.access_trace}")
    print(f"Running Hash: {dataset.running_access_hash}")
    print("------------------------------------------------------------------")

    compute_time_start = time.time()

    # # Manually going through the dataset (no parallel)
    # all_words = Counter()
    # for i in tqdm(range(len(dataset)), desc="Processing dataset"):
    #     item = dataset[i]
    #     text = item.get("text", "")
    #     if not text:
    #         continue
    #     words = word_pattern.findall(text.lower())
    #     filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    #     all_words.update(filtered_words)

    num_proc=8

    # Process using PyTorch's DataLoader
    def torch_wrapper_extract_words(dataset):
        def test_extract_words(batch):
            state = get_state()
            results = []
            for sample in batch:
                text = sample.get("text", "")
                words = word_pattern.findall(text.lower())
                filtered = [w for w in words if w not in stop_words and len(w) > 1]

                state.total_accesses = dataset.total_accesses
                state.getitem_load_time = dataset.getitem_load_time
                state.getitem_measure_time = dataset.getitem_measure_time
                state.running_access_hash = dataset.running_access_hash
                results.append({"filtered_words": filtered, "__state__": state.to_dict()})
            return results
        return test_extract_words

    ds_loader = DataLoader(dataset, collate_fn=torch_wrapper_extract_words(dataset), batch_size=1, shuffle=False, num_workers=num_proc)
    ds_loader_len = len(ds_loader)

    # Group states by process ID
    final_state = {}
    all_words = Counter()
    for batch in tqdm(ds_loader, total=ds_loader_len, desc=f"Mapping dataset (num_proc={num_proc})"):
        for record in batch: 
            all_words.update(record["filtered_words"])
            stat = record.get("__state__")
            if stat is not None:
                pid = stat["pid"]
                # Keep only the state with the highest total_accesses per PID
                if pid not in final_state or stat["total_accesses"] > final_state[pid]["total_accesses"]:
                    final_state[pid] = stat
    
    print("Mapped")
    print("All PIDs in final_state:", list(final_state.keys()))


    # Word frequency
    most_common = all_words.most_common(10000)

    # Word length frequency
    length_counter = Counter(len(word) for word in all_words)
    lengths = sorted(length_counter.keys())
    frequencies = [length_counter[l] for l in lengths]

    compute_time_end = time.time()
    compute_time = compute_time_end - compute_time_start
    
    dataset.total_accesses = sum(s["total_accesses"] for s in final_state.values()) # / num_proc
    dataset.getitem_load_time = sum(s["getitem_load_time"] for s in final_state.values()) / num_proc
    dataset.getitem_measure_time = sum(s["getitem_measure_time"] for s in final_state.values()) / num_proc

    if args.measure:
        values = list(final_state.values())
        dataset.running_access_hash.reset(value=int(values[0]["running_access_hash"]))
        if num_proc > 1:
            for s in values[1:]:
                dataset.running_access_hash.multiply(s["running_access_hash"])
        print(dataset.running_access_hash.digest())

    print(f"Compute time: {compute_time}")
    print("------------------------------------------------------------------")    

    # Plot word length distribution
    plt.figure(figsize=(10, 6))
    plt.bar(lengths, frequencies, color='skyblue')
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    plt.title(f'Word Length Distribution in {dataset_name}')
    plt.grid(axis='y')
    plt.xticks(lengths)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    # plt.savefig("word_length_distribution.png", dpi=300)
    plot_buf = io.BytesIO()
    plt.savefig(plot_buf, format='png', dpi=300)
    plot_buf.seek(0) 

    # Save all of the distribution results
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["Rank", "Word", "Frequency"])
    for i, (word, freq) in enumerate(most_common, 1):
        writer.writerow([i, word, freq])

    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    if args.measure:
        _, plot_hash, plot_output_measure_time = palm.measure_output(plot_buf.getvalue())
        _, top_words_hash, top_words_output_measure_time = palm.measure_output(csv_bytes)

    directory = f"./llm_output/{args.attestation_type}"
    directory = os.path.join(directory, f"in-memory-{args.in_memory}")
    evi_dir = os.path.join(directory, "evidence")
    if not os.path.exists(evi_dir):
        os.makedirs(evi_dir, exist_ok=True)

    plot_output_storage_time_start = time.time()
    with open(f"{directory}/word_length_distribution.png", "wb") as f:
        f.write(plot_buf.getvalue())
    plot_output_storage_time_end = time.time()
    plot_output_storage_time = plot_output_storage_time_end - plot_output_storage_time_start
    print("Saved word length distribution plot to word_length_distribution.png")

    top_words_output_storage_time_start = time.time()
    with open(f"{directory}/top_words.csv", "w", newline='') as f:
        f.write(csv_buffer.getvalue())
    top_words_output_storage_time_end = time.time()
    top_words_output_storage_time = top_words_output_storage_time_end - top_words_output_storage_time_start
    print("Saved top 10k words to top_words.csv")

    attestation_time = 0
    output_measurement_time = 0
    output_storage_time = 0
    model_path="NA"

    # Generate TDX TD quote using DCAP
    if args.measure:
        if dataset.in_memory:
            dataset_hashes_output = {
                split: {
                    fname: base64.b64encode(bytes.fromhex(h)).decode('utf-8')
                    for fname, h in dataset_hashes[split].items()
                }
                for split in dataset_hashes
            }
        else:
            h = dataset.running_access_hash.digest() 
            h_bytes = h.to_bytes((h.bit_length() + 7) // 8, byteorder='big')
            dataset_hashes_output = {
                'train': base64.b64encode(h_bytes).decode('utf-8')
            }
        payload = {
            'dataset_hash': dataset_hashes_output,
            'top_words_hash': base64.b64encode(top_words_hash).decode('utf-8'),
            'word_length_distribution_plot': base64.b64encode(plot_hash).decode('utf-8'),
        }
        with open(f'{evi_dir}/payload_distribution.json', 'w') as f:
            json.dump(payload, f)
        output_measurement_time, attestation_time, output_storage_time = palm.attest(payload, f"{evi_dir}/attestation_{args.attestation_type}.json")
    
    exp_config = init_exp_config(args, 
                    model_path=model_path, 
                    mem_usage=mem_usage, 
                    total_access=dataset.total_accesses, 
                    compute_time=compute_time, 
                    input_dataset_load_time=dataset.load_time, 
                    input_dataset_measure_time=dataset.measure_time,
                    getitem_load_time=dataset.getitem_load_time,
                    getitem_measure_time=dataset.getitem_measure_time,
                    input_model_load_time=0,
                    input_model_measure_time=0,
                    attestation_time=attestation_time,
                    output_dataset_storage_time=0,
                    output_dataset_measurement_time=0,
                    output_model_storage_time=0,
                    output_model_measurement_time=0,
                    output_storage_time=(output_storage_time + top_words_output_storage_time + plot_output_storage_time),
                    output_measurement_time=(output_measurement_time + top_words_output_measure_time + plot_output_measure_time),
                    )

    return exp_config

# Proof of binding
def dataset_binding_attestation(args):
    
    dataset_name = "bookcorpus"
    path = os.path.join("./data/", dataset_name)
    dataset_path = os.path.join(path, "dataset")
    while True:
        try:
            print("Loading dataset...")
            in_process, mem_before_proc, mem_before_sys = start_memory_measure()

            # Load and measure in-memory dataset
            ds_in_mem, ds_in_mem_hashes = palm.load_dataset(load_path=path, load_in_memory=True, measure=True)
            mem_usage = end_memory_measure(in_process, mem_before_proc, mem_before_sys)

            # Initialize memory-mapped dataset
            ds_mem_mapped, _ = palm.load_dataset(load_path=path, load_in_memory=False, measure=True)
            break
        except Exception as e:
            print(f"Fallback to HF download: {e}")
            dataset = load_dataset(dataset_name, split="train")
            dataset.save_to_disk(dataset_path)
            dataset.cleanup_cache_files()
            print("Saved dataset to disk.")

    num_proc=8

    def wrapper_dataset_passthrough(dataset):
        def dataset_passthrough(batch):
            state = get_state()
            results = []
            for _ in batch:
                state.total_accesses = dataset.total_accesses
                state.getitem_load_time = dataset.getitem_load_time
                state.getitem_measure_time = dataset.getitem_measure_time
                state.running_access_hash = dataset.running_access_hash
                results.append({"__state__": state.to_dict()})
            return results
        return dataset_passthrough

    ds_loader = DataLoader(ds_mem_mapped, collate_fn=wrapper_dataset_passthrough(ds_mem_mapped), batch_size=1, shuffle=False, num_workers=num_proc)
    ds_loader_mapped_len = len(ds_loader)
    ds_in_mem_len = len(ds_in_mem)

    compute_start = time.time()

    # Group states by process ID
    final_state = {}
    for batch in tqdm(ds_loader, total=ds_loader_mapped_len, desc=f"Processing through dataset (num_proc={num_proc})"):
        for record in batch: 
            stat = record.get("__state__")
            if stat is not None:
                pid = stat["pid"]
                # Keep only the state with the highest total_accesses per PID
                if pid not in final_state or stat["total_accesses"] > final_state[pid]["total_accesses"]:
                    final_state[pid] = stat

    compute_end = time.time()
    compute_time = (compute_end - compute_start) 
    print("Memory-mapped dataset compute time:", compute_time)
    
    print("Mapped")

    ds_mem_mapped.total_accesses = sum(s["total_accesses"] for s in final_state.values())
    ds_mem_mapped.getitem_load_time = sum(s["getitem_load_time"] for s in final_state.values()) / num_proc
    ds_mem_mapped.getitem_measure_time = sum(s["getitem_measure_time"] for s in final_state.values()) / num_proc

    values = list(final_state.values())
    ds_mem_mapped.running_access_hash.reset(value=int(values[0]["running_access_hash"]))

    if num_proc > 1:
        for s in values[1:]:
            ds_mem_mapped.running_access_hash.multiply(s["running_access_hash"])

    print("Memory-mapped")
    print(f"Total accesses: {ds_mem_mapped.total_accesses}")
    print(f"Total load time: {ds_mem_mapped.getitem_load_time:.6f} s")
    print(f"Total measure time: {ds_mem_mapped.getitem_measure_time:.6f} s")
    print(ds_mem_mapped.running_access_hash.digest())
    print("------------------------------------------------------------------")
    print("In-memory")
    print(f"Total accesses: {ds_in_mem.total_accesses}")
    print(f"Total load time: {ds_in_mem.getitem_load_time:.6f} s")
    print(f"Total measure time: {ds_in_mem.getitem_measure_time:.6f} s")
    print(ds_in_mem_hashes)

    directory = f"./llm_output/{args.attestation_type}"
    evi_dir = os.path.join(directory, "evidence")
    if not os.path.exists(evi_dir):
        os.makedirs(evi_dir, exist_ok=True)

    attestation_time = 0
    output_measurement_time = 0
    output_storage_time = 0
    model_path = "NA"

    # Generate TDX TD quote using DCAP
    if args.measure:
        assert ds_mem_mapped.total_accesses == ds_in_mem_len, (
            f"Mismatch: total_accesses={ds_mem_mapped.total_accesses}, "
            f"expected={ds_in_mem_len}"
        )

        # In-memory dataset
        dataset_in_mem_hashes_output = {
            split: {
                fname: base64.b64encode(bytes.fromhex(h)).decode('utf-8')
                for fname, h in ds_in_mem_hashes[split].items()
            }
            for split in ds_in_mem_hashes
        }
        # Memory-mapped dataset
        h = ds_mem_mapped.running_access_hash.digest() 
        h_bytes = h.to_bytes((h.bit_length() + 7) // 8, byteorder='big')
        dataset_memory_mapped_hashes_output = {
            'train': base64.b64encode(h_bytes).decode('utf-8')
        }
        payload = {
            'dataset_in_memory_hash': dataset_in_mem_hashes_output,
            'dataset_memory_mapped_hash': dataset_memory_mapped_hashes_output,
        }
        with open(f'{evi_dir}/payload_dataset_binding.json', 'w') as f:
            json.dump(payload, f)
        output_measurement_time, attestation_time, output_storage_time = palm.attest(payload, f"{evi_dir}/attestation_{args.attestation_type}.json")
    
    exp_config = init_exp_config(args, 
                    model_path=model_path, 
                    mem_usage=mem_usage, 
                    total_access=(ds_mem_mapped.total_accesses), 
                    compute_time=compute_time, 
                    input_dataset_load_time=(ds_in_mem.load_time), 
                    input_dataset_measure_time=(ds_in_mem.measure_time),
                    getitem_load_time=(ds_mem_mapped.getitem_load_time),
                    getitem_measure_time=(ds_mem_mapped.getitem_measure_time),
                    input_model_load_time=0,
                    input_model_measure_time=0,
                    attestation_time=attestation_time,
                    output_dataset_storage_time=0,
                    output_dataset_measurement_time=0,
                    output_model_storage_time=0,
                    output_model_measurement_time=0,
                    output_storage_time=(output_storage_time),
                    output_measurement_time=(output_measurement_time),
                    )

    return exp_config
    
# Proof of dataset preprocessing
def dataset_preprocess_attestation(args):
    
    path = "./data/bookcorpus"
    dataset_path = os.path.join(path, "dataset")

    tokenizer_load_time_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer_load_time_end = time.time()
    tokenizer_load_time = tokenizer_load_time_end - tokenizer_load_time_start
    tokenizer.pad_token=tokenizer.eos_token

    # Measure the tokenizer used in the operation
    if args.measure:
        tokenizer_measurement, tokenizer_measure_time = palm.save_tokenizer(tokenizer, "./", save_to_disk=False, measure=args.measure)

    while True:
        try:
            in_process, mem_before_proc, mem_before_sys = start_memory_measure()
            dataset, dataset_hashes = palm.load_dataset(load_path=path, load_in_memory=args.in_memory, measure=args.measure)
            mem_usage = end_memory_measure(in_process, mem_before_proc, mem_before_sys)
            break
        except Exception as e:
            print("Dataset not found...")
            dataset = load_dataset("bookcorpus")
            dataset.save_to_disk(dataset_path)
    
    compute_time_start = time.time()

    
    def torch_wrapper_tokenize(dataset, tokenizer):
        def process_batch(batch):
            state = get_state()
            results = []
            for sample in batch:
                text = sample.get("text", "")
                tokenized = tokenizer(text=text)

                state.total_accesses = dataset.total_accesses
                state.getitem_load_time = dataset.getitem_load_time
                state.getitem_measure_time = dataset.getitem_measure_time
                state.running_access_hash = dataset.running_access_hash

                results.append({"tokenized": tokenized, "__state__": state.to_dict()})
            return results
        return process_batch
    
    num_proc=8

    ds_loader = DataLoader(
        dataset,
        collate_fn=torch_wrapper_tokenize(dataset, tokenizer),
        batch_size=1,
        shuffle=False,
        num_workers=num_proc,
    )
    ds_loader_len = len(ds_loader)

    final_state = {}
    tokenized_records = []

    for batch in tqdm(ds_loader, total=ds_loader_len, desc=f"Tokenizing dataset (num_proc={num_proc})"):
        for record in batch:
            tokenized_records.append(record["tokenized"])
            stat = record.get("__state__")
            if stat is not None:
                pid = stat["pid"]
                if pid not in final_state or stat["total_accesses"] > final_state[pid]["total_accesses"]:
                    final_state[pid] = stat

    print("Tokenization complete.")
    print("All PIDs in final_state:", list(final_state.keys()))

    dataset.total_accesses = sum(s["total_accesses"] for s in final_state.values())
    dataset.getitem_load_time = sum(s["getitem_load_time"] for s in final_state.values()) / num_proc
    dataset.getitem_measure_time = sum(s["getitem_measure_time"] for s in final_state.values()) / num_proc
    print(dataset.getitem_measure_time)

    if hasattr(dataset, "running_access_hash"):
        values = list(final_state.values())
        dataset.running_access_hash.reset(value=int(values[0]["running_access_hash"]))
        if num_proc > 1:
            for s in values[1:]:
                dataset.running_access_hash.multiply(s["running_access_hash"])
        print("Updated hash:", dataset.running_access_hash.digest())

    tokenized_ds = Dataset.from_list(tokenized_records)
    concated_ds = tokenized_ds.map(utils.concat,batched=True,batch_size=1000000,num_proc=8)

    chunked_ds = concated_ds.map(utils.chunk,batched=True,batch_size=2,num_proc=8)
    compute_time_end = time.time()

    compute_time = compute_time_end - compute_time_start
    print("Compute time:", compute_time)

    directory = f"./llm_output/{args.attestation_type}"
    directory = os.path.join(directory, f"in-memory-{args.in_memory}")
    evi_dir = os.path.join(directory, "evidence")
    if not os.path.exists(evi_dir):
        os.makedirs(evi_dir, exist_ok=True)

    # Save and measure preprocessed dataset
    output_dataset_storage_time_start = time.time()
    results, output_dataset_measure_time = palm.save_dataset(chunked_ds, directory)
    output_dataset_storage_time_end = time.time()
    output_dataset_storage_time = (output_dataset_storage_time_end - output_dataset_storage_time_start) - output_dataset_measure_time

    attestation_time = 0
    output_measurement_time = 0
    output_storage_time = 0
    model_path="NA"

    if args.measure:
        print("--------------------------------------------")
        if dataset.in_memory:
            dataset_hashes_output = {
                split: {
                    fname: base64.b64encode(bytes.fromhex(h)).decode('utf-8')
                    for fname, h in dataset_hashes[split].items()
                }
                for split in dataset_hashes
            }
        else:
            h_train = dataset.running_access_hash.digest() 
            h_train_bytes = h_train.to_bytes((h_train.bit_length() + 7) // 8, byteorder='big')
            dataset_hashes_output = {
                'train': base64.b64encode(h_train_bytes).decode('utf-8'),
            }
        payload = {
            'tokenizer_hash': {
                fname: base64.b64encode(h).decode('utf-8')
                for fname, (_, h, _) in tokenizer_measurement.items()
            },
            'dataset_hash': {
                "hf_ds": {
                    split: {
                        os.path.basename(shard["file"]): base64.b64encode(shard["hash"]).decode("utf-8")
                        for shard in shard_list if shard["hash"] is not None
                    }
                    for split, shard_list in results.items()
                },
                'original_dataset_hashes': dataset_hashes_output
            },
        }
        with open(f'{evi_dir}/payload_pretrain.json', 'w') as f:
            json.dump(payload, f)
        output_measurement_time, attestation_time, output_storage_time = palm.attest(payload, f"{evi_dir}/attestation_{args.attestation_type}.json")

    exp_config = init_exp_config(args, 
                    model_path=model_path, 
                    mem_usage=mem_usage, 
                    total_access=(dataset.total_accesses), 
                    compute_time=compute_time, 
                    input_dataset_load_time=dataset.load_time, 
                    input_dataset_measure_time=dataset.measure_time,
                    getitem_load_time=dataset.getitem_load_time,
                    getitem_measure_time=dataset.getitem_measure_time,
                    input_model_load_time=tokenizer_load_time,
                    input_model_measure_time=tokenizer_measure_time,
                    attestation_time=attestation_time,
                    output_dataset_storage_time=output_dataset_storage_time,
                    output_dataset_measurement_time=output_dataset_measure_time,
                    output_model_storage_time=0,
                    output_model_measurement_time=0,
                    output_storage_time=(output_storage_time),
                    output_measurement_time=(output_measurement_time),
                    )

    return exp_config


def handle_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device",type=str,default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),help="GPU ID for this process")
    parser.add_argument("--model",type=str,default="llama",help="Finetuning: [llama, gemma, phi], Inference: [llama, gemma, phi]")
    parser.add_argument("--model_size",type=str,default="L",help="[One of: [S, L]")
    parser.add_argument("--attestation_type",type=str, default="pretrain", help="One of: [pretrain, finetune, inference, eval, eval_bleu, quantize, distribution, bind, preprocess]")
    parser.add_argument("--measure", action="store_true", help="TD quote generation and measurement (default: False)")
    parser.add_argument("--exp_id",type=int, default=0, help="For reporting purposes.")
    parser.add_argument("--in_memory", action="store_true", help="Use in-memory Dataset (default: False)")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = handle_args()

    print("Starting")


    if args.attestation_type == "finetune":
        exp_config = finetuning_attestation(args)
    elif args.attestation_type == "inference":
        exp_config = inference_attestation(args)
    elif args.attestation_type == "eval":
        exp_config = evaluation_attestation(args)
    elif args.attestation_type == "eval_bleu":
        exp_config = eval_bleu(args)
    elif args.attestation_type == "pretrain":
        exp_config = pretraining_attestation(args)
    elif args.attestation_type == "distribution":
        exp_config = distribution_attestation(args)
    elif args.attestation_type == "quantize":
        exp_config = quantization_attestation(args)
    elif args.attestation_type == "bind":
        exp_config = dataset_binding_attestation(args)
    elif args.attestation_type == "preprocess":
        exp_config = dataset_preprocess_attestation(args)
    else:
        print("Incorrect attestation type")
        exit()

    row_data = {
        "attestation_type": exp_config["attestation_type"],
        "model_path": exp_config["model_path"],
        "measure": exp_config["measure"],
        "in_memory": exp_config["in_memory"],
        "mem_usage": exp_config["mem_usage"],
        "total_access": exp_config["total_access"],
        "compute_time": exp_config["compute_time"],

        "input_dataset_load_time": exp_config["input"]["input_dataset_load_time"],
        "input_dataset_measure_time": exp_config["input"]["input_dataset_measure_time"],
        "getitem_load_time": exp_config["input"]["getitem_load_time"],
        "getitem_measure_time": exp_config["input"]["getitem_measure_time"],
        "input_model_load_time": exp_config["input"]["input_model_load_time"],
        "input_model_measure_time": exp_config["input"]["input_model_measure_time"],

        "attestation_time": exp_config["output"]["attestation_time"],
        "output_dataset_storage_time": exp_config["output"]["output_dataset_storage_time"],
        "output_dataset_measurement_time": exp_config["output"]["output_dataset_measurement_time"],
        "output_model_storage_time": exp_config["output"]["output_model_storage_time"],
        "output_model_measurement_time": exp_config["output"]["output_model_measurement_time"],
        "output_total_storage_time": exp_config["output"]["output_total_storage_time"],
        "output_total_measure_time": exp_config["output"]["output_total_measure_time"],
    }

    csv_path = "./llm_results.csv"

    df = pd.DataFrame([row_data])
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"Results written to {csv_path}")
