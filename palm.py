import copy
import json 
import os
from tqdm import tqdm
import time
import io
import hashlib

from peft import get_peft_model_state_dict
from safetensors.torch import save
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from datasets import DatasetDict, DatasetInfo, SplitInfo, concatenate_datasets, load_dataset, load_from_disk
import pyarrow as pa
import pyarrow.ipc as ipc

import quote_generator
from msh import *

import utils
from measured_file_read import *

class MeasureDataset(Dataset):
    def __init__(self, dataset, load_in_memory, data_load_time=0, data_measure_time=0, measure=True):
        
        self.dataset = dataset
        self.in_memory = load_in_memory

        if self.in_memory:
            self.load_time = data_load_time
            self.measure_time = data_measure_time
        else:
            self.load_time = 0
            self.measure_time = 0

        self.total_accesses = 0
        self.getitem_load_time = 0
        self.getitem_measure_time = 0
        self.measure = measure
        self.access_trace = []
        self.running_access_hash = MSetMuHash()

    @property
    def _info(self):
        return self.dataset._info

    def __repr__(self):
        return f"MeasureDataset(\n{repr(self.dataset)}\n)"

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __getitem__(self, idx):
        self.total_accesses += 1
        # print("__getitem__ here!")
        # print(self.total_accesses)
        record_load_time_start = time.time()
        item = self.dataset[idx]
        record_load_time_end = time.time()
        diff = (record_load_time_end - record_load_time_start)
        self.getitem_load_time += diff
        if not self.in_memory:
            if self.measure:
                # self.access_trace.append(idx)
                record_measure_time_start = time.time()
                self.running_access_hash.add(json.dumps(item))
                record_measure_time_end = time.time()
                self.getitem_measure_time += (record_measure_time_end - record_measure_time_start)
                # print(self.running_access_hash.digest())
        # print(self.total_accesses)
        return item

    def set_measure(self, measure):
        self.measure = measure
    
    def select(self, *args, **kwargs):
        return MeasureDataset(
            self.dataset.select(*args, **kwargs),
            load_in_memory=self.in_memory,
            data_load_time=self.load_time,
            data_measure_time=self.measure_time,
            measure=self.measure,
        )
    
    def map(self, *args, **kwargs):
        # mapped_dataset = self.dataset.map(*args, **kwargs)
        return MeasureDataset(
            self.dataset.map(*args, **kwargs),
            load_in_memory=self.in_memory,
            data_load_time=self.load_time,
            data_measure_time=self.measure_time,
            measure=self.measure,
        )

    def save_to_disk(self, path):
        self.dataset.save_to_disk(path)

    def __del__(self):
        # Safely try to delete the inner dataset if it exists
        try:
            if hasattr(self, "dataset") and hasattr(self.dataset, "__del__"):
                self.dataset.__del__()
        except Exception:
            pass  

    def __getattr__(self, name):
        return getattr(self.dataset, name)

def load_dataset(load_path=None, load_in_memory=False, split=None, measure=False):
    if load_path is None:
        raise ValueError("You must provide either a dataset or a load_path.")
    dataset_hashes = {}
    split_load_times = {}
    split_measure_times = {}

    if os.path.exists(load_path):
        # Local dataset
        if load_in_memory:
            dataset, dataset_hashes, split_load_times, split_measure_times = load_dataset_from_disk(load_path, split=split)
            print("Loaded from disk (in-memory)")
        else:
            if split:
                split_path = os.path.join(load_path, split)
                if not os.path.exists(split_path):
                    raise FileNotFoundError(f"Split '{split}' not found in {load_path}")
                dataset = load_from_disk(split_path)
            else:
                dataset = load_from_disk(load_path)
            print("Loaded from disk (memory-mapped).")
    else:
        dataset = load_dataset(load_path, split=split)
        print("Loaded from huggingface (memory-mapped).")
    
    def wrap(split_name, split_ds):
        load_time = split_load_times.get(split_name, 0.0)
        measure_time = split_measure_times.get(split_name, 0.0)
        return MeasureDataset(split_ds, load_in_memory, load_time, measure_time, measure=measure)

    if isinstance(dataset, dict):
        if split is not None or len(dataset) == 1:
            key = list(dataset.keys())[0]
            dataset = wrap(key, dataset[key])
        else:
            dataset = {k: wrap(k, v) for k, v in dataset.items()}
    else:
        dataset = wrap('train', dataset)

    return dataset, dataset_hashes

def load_dataset_from_disk(path, split=None):
    dataset_dict = {}
    hashes = {}
    split_load_times = {}
    split_measure_times = {}

    has_subdirs = any(os.path.isdir(os.path.join(path, split)) for split in ['train', 'test'])
    all_splits = ['train', 'test'] if has_subdirs else ['']

    # Filter only the requested split
    if split is not None:
        if split not in all_splits:
            raise ValueError(f"Requested split '{split}' not found in dataset.")
        splits_to_process = [split]
    else:
        splits_to_process = all_splits

    print("Loading:", path)
    for split in splits_to_process:
        split_key = split or 'train'
        split_path = os.path.join(path, split) if split else path
        state_path = os.path.join(split_path, "state.json")

        if not os.path.exists(state_path):
            continue

        with open(state_path, "r") as f:
            state = json.load(f)
        arrow_files = [entry["filename"] for entry in state.get("_data_files", [])]

        split_datasets = []
        split_hashes = {}
        split_total_load_time = 0
        split_total_measure_time = 0

        for fname in tqdm(arrow_files, desc=f"Loading {split or 'default'} split"):
            full_path = os.path.join(split_path, fname)
            buffer_ds, chunk_hash, load_time, measure_time = open_measured(full_path, "rb")

            # Check file extension for Arrow dataset
            if full_path.endswith(".arrow"):
                try:
                    buffer_ds = pa.py_buffer(buffer_ds.getvalue())
                except Exception as e:
                    raise RuntimeError(f"Failed to read Arrow file '{path}': {e}")
                
            ds = HFDataset.from_buffer(buffer_ds)
            split_total_load_time += load_time
            split_total_measure_time += measure_time
            split_datasets.append(ds)
            split_hashes[fname] = chunk_hash

        if split_datasets:
            dataset_dict[split_key] = concatenate_datasets(split_datasets)
            hashes[split_key] = split_hashes
            split_load_times[split_key] = split_total_load_time
            split_measure_times[split_key] = split_total_measure_time

    return DatasetDict(dataset_dict), hashes, split_load_times, split_measure_times

def default_chat_template(conversation):
    return utils.default_chat_template(conversation)

def get_tokenizer_config_dict(tokenizer, save_directory="/tmp/palm", filename_prefix="", save_jinja_files=True):
    if hasattr(tokenizer, "tokenizer"):
        base_tokenizer = tokenizer.tokenizer
    else:
        base_tokenizer = tokenizer

    try:
        tokenizer_config = copy.deepcopy(base_tokenizer.init_kwargs)
        target_keys = set(base_tokenizer.init_kwargs.keys())
    except AttributeError:
        print("[Warning] Tokenizer has no init_kwargs. Returning empty config.")
        return {}

    target_keys.update(["model_max_length", "clean_up_tokenization_spaces"])
    for k in target_keys:
        if hasattr(base_tokenizer, k):
            tokenizer_config[k] = getattr(base_tokenizer, k)

    try:
        tokenizer_config.update(base_tokenizer.special_tokens_map)
    except AttributeError:
        print("[Info] No special_tokens_map in tokenizer.")

    if not tokenizer_config.get("extra_special_tokens") and hasattr(base_tokenizer, "extra_special_tokens"):
        tokenizer_config["extra_special_tokens"] = base_tokenizer.extra_special_tokens

    if hasattr(base_tokenizer, "get_chat_template"):
        try:
            chat_template = base_tokenizer.get_chat_template()
            if chat_template is not None:
                tokenizer_config["chat_template"] = chat_template
        except Exception as e:
            print(f"Skipping chat_template: {e}")

    if hasattr(base_tokenizer, "init_inputs") and base_tokenizer.init_inputs:
        tokenizer_config["init_inputs"] = copy.deepcopy(base_tokenizer.init_inputs)
    if hasattr(base_tokenizer, "vocab_files_names"):
        for file_id in base_tokenizer.vocab_files_names.keys():
            tokenizer_config.pop(file_id, None)
    if hasattr(base_tokenizer, "convert_added_tokens"):
        tokenizer_config = base_tokenizer.convert_added_tokens(
            tokenizer_config, add_type_field=True, save=True
        )
    if hasattr(base_tokenizer, "added_tokens_decoder"):
        tokenizer_config["added_tokens_decoder"] = {
            k: v.__getstate__() for k, v in base_tokenizer.added_tokens_decoder.items()
        }

    tokenizer_class = base_tokenizer.__class__.__name__
    if tokenizer_class.endswith("Fast") and getattr(base_tokenizer, "can_save_slow_tokenizer", False):
        tokenizer_class = tokenizer_class[:-4]
    tokenizer_config["tokenizer_class"] = tokenizer_class

    if getattr(base_tokenizer, "_auto_map", None) is not None:
        tokenizer_config["auto_map"] = base_tokenizer._auto_map
    if getattr(base_tokenizer, "_processor_class", None) is not None:
        tokenizer_config["processor_class"] = base_tokenizer._processor_class

    if getattr(base_tokenizer, "_auto_class", None) is not None:
        try:
            from transformers.utils.hub import custom_object_save
            custom_object_save(base_tokenizer, save_directory, config=tokenizer_config)
        except Exception as e:
            print(f"[Warning] Could not save custom object: {e}")

    for key in ["name_or_path", "special_tokens_map_file", "tokenizer_file", "device_map"]:
        tokenizer_config.pop(key, None)

    return tokenizer_config

def obj_to_byte(obj):
    if isinstance(obj, bytes):
        bytes_object = obj
    else:
        bytes_object = json.dumps(obj, indent=2, default=str).encode("utf-8")
        
    bytes_object = io.BytesIO(bytes_object)

    try:
        bytes_object = bytes_object.getvalue()
    except Exception as e:
        print(".getvalue() doesn't work!")
        bytes_object = bytes_object
    
    return bytes_object

def measure_output(bytes_object):
    if not isinstance(bytes_object, bytes):
        bytes_object = obj_to_byte(bytes_object)
    
    hasher = hashlib.sha256()
    measure_start = time.time()
    hasher.update(bytes_object)
    h = hasher.digest()
    measure_end = time.time()
    measure_time = measure_end - measure_start

    return bytes_object, h, measure_time

def get_tokenizer_files(tokenizer):
    results = {}
    base_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    # tokenizer_config.json
    config_dict = get_tokenizer_config_dict(tokenizer)
    results["tokenizer_config.json"] = config_dict

    # tokenizer.json (Fast tokenizer full model)
    if hasattr(base_tokenizer, "backend_tokenizer") and base_tokenizer.backend_tokenizer is not None:
        try:
            tokenizer_dict = json.loads(base_tokenizer.backend_tokenizer.to_str())
            tokenizer_dict.setdefault("added_tokens", [])
            results["tokenizer.json"] = tokenizer_dict
        except Exception as e:
            print(f"[Warning] Failed to read backend_tokenizer: {e}")

    # special_tokens_map.json
    try:
        stm_dict = base_tokenizer.special_tokens_map_extended
        stm_dict = utils.config_to_string(stm_dict)
        results["special_tokens_map.json"] = stm_dict
    except AttributeError:
        print("[Info] No special_tokens_map_extended found.")

    # added_tokens.json
    if hasattr(base_tokenizer, "added_tokens_encoder") and base_tokenizer.added_tokens_encoder:
        added_dict = dict(sorted(base_tokenizer.added_tokens_encoder.items()))
        results["added_tokens.json"] = added_dict

    # vocab.json
    if getattr(base_tokenizer, "vocab", None):
        vocab_dict = dict(sorted(base_tokenizer.vocab.items()))
        results["vocab.json"] = vocab_dict

    # merges.txt
    if hasattr(base_tokenizer, "merges"):
        merges_text = "\n".join(base_tokenizer.merges)
        results["merges.txt"] = merges_text.encode("utf-8")

    return results

def measure_tokenizer_files(files_dict, measure=True):
    """Return dict -> (bytes, hash, time)"""
    results = {}
    total_time = 0

    for filename, content in files_dict.items():
        if measure:
            file_bytes, h, t = measure_output(content)
        else:
            h = None
            t = 0
            file_bytes = obj_to_byte(content)
        results[filename] = (file_bytes, h, t)
        total_time += t

    return results, total_time

def save_model(model, directory="./", save_to_disk=True, measure=True):
    save_adapter = utils.is_adapter_model(model)
    if save_adapter:
        model_path = os.path.join(directory, "adapter_model.safetensors")
        config_path = os.path.join(directory, "adapter_config.json")
    else:
        model_path = os.path.join(directory, "model.safetensors")
        config_path = os.path.join(directory, "config.json")
    
    if save_to_disk:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    model_hash, model_measure_time = save_model_safetensors(model, model_path, save_adapter, save_to_disk, measure)
    config_hash, config_measure_time = save_model_config(model, config_path, save_adapter, save_to_disk, measure)

    return model_hash, config_hash, (model_measure_time + config_measure_time)

def save_model_config(model, config_path, save_adapter, save_to_disk, measure):
    if save_adapter:
        config_dict = model.peft_config["default"].to_dict()
    else:
        config_dict = model.config.to_dict()
        
    config_dict = utils.config_to_string(config_dict)
    config_dict = dict(sorted(config_dict.items()))
    
    if measure:
        config_bytes, config_hash, config_measure_time = measure_output(config_dict)
        print(f"Config measure time: {config_measure_time}s")
    else:
        config_hash = None
        config_measure_time = 0
        config_bytes = obj_to_byte(config_dict)

    if save_to_disk:
        with open(config_path, "wb") as f:
            f.write(config_bytes)
        # print("Saved to:", config_path)

    return config_hash, config_measure_time

def save_model_safetensors(model, model_path, save_adapter, save_to_disk, measure):
    # print(save_adapter)
    if save_adapter:
        state_dict = get_peft_model_state_dict(model)
    else:
        state_dict = model.state_dict()
    # state_dict = model.state_dict()
    try:
        print("Measuring...")
        bytes_object = save(state_dict, metadata={"format": "pt"})
    except Exception as e:
        print("Some tensors are shared-memory. Saving with another method")
        for key, tensor in tqdm(state_dict.items(), desc="Cloning shared tensors", total=len(state_dict)):
            if tensor.is_shared():
                state_dict[key] = tensor.clone().detach()
        print("Measuring...")
        bytes_object = save(state_dict, metadata={"format": "pt"})
    
    if measure:
        model_bytes, model_hash, model_measure_time = measure_output(bytes_object)
        print(f"Model measure time: {model_measure_time}s")
    else:
        model_hash = None
        model_measure_time = 0
        model_bytes = obj_to_byte(bytes_object)

    if save_to_disk:
        with open(model_path, 'wb') as f:
            f.write(model_bytes)
        # print("Saved to:", model_path)
        
    return model_hash, model_measure_time

def save_tokenizer(tokenizer, tokenizer_path, save_to_disk=True, measure=True):

    tokenizer_dict = get_tokenizer_files(tokenizer)
    tokenizer_results, tokenizer_measure_time = measure_tokenizer_files(tokenizer_dict, measure)

    if save_to_disk:
        os.makedirs(tokenizer_path, exist_ok=True)
        for filename in sorted(tokenizer_results.keys()):
            data_bytes, _, _ = tokenizer_results[filename]
            path = os.path.join(tokenizer_path, filename)
            with open(path, "wb") as f:
                f.write(data_bytes)
            # print("Saved to:", data_bytes)
    
    return tokenizer_results, tokenizer_measure_time

def save_dataset(dataset, dataset_path, max_shard_size_mb=500, measure=True):

    os.makedirs(dataset_path, exist_ok=True)
    all_results = {}

    total_measure_time = 0

    # Determine splits
    if isinstance(dataset, DatasetDict):
        splits = dataset.keys()
        is_single_dataset = False
    elif isinstance(dataset, dict):
        splits = dataset.keys()
        is_single_dataset = False
    else:  # single Dataset
        splits = ["train"]
        dataset = {"train": dataset}
        is_single_dataset = True

    for split_name in splits:
        split_ds = dataset[split_name]

        # unwrap MeasureDataset dict if needed
        if isinstance(split_ds, dict):
            if len(split_ds) != 1:
                raise ValueError(f"Expected single Dataset in dict for split '{split_name}', got {split_ds.keys()}")
            split_ds = list(split_ds.values())[0]

        if is_single_dataset:
            split_folder = dataset_path
        else:
            split_folder = os.path.join(dataset_path, split_name)
            os.makedirs(split_folder, exist_ok=True)

        schema = split_ds.data.schema
        max_shard_bytes = max_shard_size_mb * 1024 * 1024  # MB -> bytes

        # Check shard boundaries
        shard_batches = []
        current_shard_batches = []
        current_shard_bytes = 0

        for batch in split_ds.data.to_batches():
            batch_bytes = io.BytesIO()
            with ipc.new_stream(batch_bytes, batch.schema) as temp_writer:
                temp_writer.write_batch(batch)
            batch_bytes = batch_bytes.getvalue()

            if current_shard_bytes + len(batch_bytes) > max_shard_bytes and current_shard_batches:
                shard_batches.append(current_shard_batches)
                current_shard_batches = []
                current_shard_bytes = 0

            current_shard_batches.append(batch)
            current_shard_bytes += len(batch_bytes)

        if current_shard_batches:
            shard_batches.append(current_shard_batches)

        num_shards = len(shard_batches)
        shard_files = []

        all_results[split_name] = []

        # Write dataset to disk
        for idx, batches in tqdm(enumerate(shard_batches), total=num_shards, desc=f"Saving {split_name} shards"):
            shard_file = os.path.join(split_folder, f"data-{idx:05d}-of-{num_shards:05d}.arrow")
            shard_files.append(shard_file)
            accumulated_bytes_io = io.BytesIO()
            shard_row_count = 0

            with pa.OSFile(shard_file, 'wb') as f:
                with ipc.new_stream(f, schema) as writer:
                    for batch in batches:
                        writer.write_batch(batch)

                        # accumulate for measurement
                        batch_bytes = io.BytesIO()
                        with ipc.new_stream(batch_bytes, batch.schema) as temp_writer:
                            temp_writer.write_batch(batch)
                        accumulated_bytes_io.write(batch_bytes.getvalue())

                        shard_row_count += batch.num_rows

            # measure shard
            if measure:
                _, shard_hash, shard_measure_time = measure_output(accumulated_bytes_io.getvalue())
            else:
                shard_hash = None
                shard_measure_time = 0
            total_measure_time += shard_measure_time
            all_results[split_name].append({
                "shard_idx": idx,
                "rows": shard_row_count,
                "hash": shard_hash,
                "time": shard_measure_time,
                "file": shard_file,
            })


        # Save dataset_info.json
        dataset_info_path = os.path.join(split_folder, "dataset_info.json")
        info_dict = utils.asdict(split_ds.info)
        info_dict = {k: info_dict[k] for k in sorted(info_dict)}
        with open(dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(info_dict, f, indent=2, ensure_ascii=False)

        # Save state.json
        state_path = os.path.join(split_folder, "state.json")
        state = {
            "_data_files": [{"filename": os.path.basename(f)} for f in shard_files],
            "_fingerprint": split_ds._fingerprint,
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": split_name
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    dataset_dict_path = os.path.join(dataset_path, "dataset_dict.json")
    with open(dataset_dict_path, "w", encoding="utf-8") as f:
        json.dump({"splits": list(splits)}, f, ensure_ascii=False)

    return all_results, total_measure_time

def attest(payload, output_filename="./output_attestation.json"):
    return quote_generator.attest(payload, output_filename)

