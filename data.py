import os
import time 
import torch
import json
import random
import posixpath
import hashlib
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from collections import Counter
import re
from torch.utils.data import Dataset, TensorDataset, Sampler, SequentialSampler
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from datasets import Dataset as HFDataset
from transformers import Trainer
from torch.utils.data import DataLoader
import nltk
from nltk.corpus import stopwords 
nltk.download('stopwords')

from itertools import chain
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.ipc as ipc
from torch.utils.data import IterableDataset
from typing import Iterator, Dict, Optional, Callable

from measured_file_read import open_measured

from msh import *

def process_cifar():
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])

    trainset = torchvision.datasets.CIFAR10(root="./data/", train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root="./data/", train=False, download=True, transform=transform_test)

    return trainset, testset

def process_utkface(path="./data/utkface.csv"):
    total_data_measure_start = time.time()
    measured_data_bytes, _, load_time = open_measured(path, 'rb')
    total_data_measure_end = time.time()
    total_data_measure_time = total_data_measure_end - total_data_measure_start
    data_measure_time = total_data_measure_time - load_time

    preprocess_start = time.time()
    df = pd.read_csv(measured_data_bytes, na_values="NA", index_col=None, sep=",", header=0)
    df['pixels']= df['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32"))
    df['pixels'] = df['pixels'].apply(lambda x: x/255)
    df['pixels'] = df['pixels'].apply(lambda x:  np.reshape(x, (1, 48,48)))
    df['pixels'] = df['pixels'].apply(lambda x: np.repeat(x, 3, axis=0))
    
    df["age"] = df["age"] > 30
    df["age"] = df["age"].astype(int)

    df["race"] = df["ethnicity"]
    df["race"] = df["race"] == 0
    df["race"] = df["race"].astype(int)   

    X = df['pixels'].to_frame()
    attr = df[ ["age", "gender", "race" ]]

    X_np = np.stack( X["pixels"].to_list() )
    attr_np = attr.to_numpy()


    X_train, X_test, attr_train, attr_test = train_test_split(X_np, attr_np, test_size=0.5, stratify=attr_np, random_state=0)
    target_index = 0
    sensitive_index = 1
    y_train = attr_train[:,target_index]
    Z_train = attr_train[:]
    y_test = attr_test[:,target_index]
    Z_test = attr_test[:]

    train_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_train)).type(torch.FloatTensor), torch.from_numpy(np.array(y_train)).type(torch.LongTensor), torch.from_numpy(np.array(Z_train)).type(torch.FloatTensor))
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_test)).type(torch.FloatTensor), torch.from_numpy(np.array(y_test)).type(torch.LongTensor), torch.from_numpy(np.array(Z_test)).type(torch.FloatTensor))
    preprocess_end = time.time()
    preprocess_time = preprocess_end - preprocess_start

    return train_data, test_data, measured_data_bytes.hasher.digest(), data_measure_time, load_time, preprocess_time


def process_census(path="./data/adult.data"):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num','martial_status', 'occupation', 'relationship', 'race', 'sex','capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']

    total_data_measure_start = time.time()
    measured_data_bytes, _, load_time = open_measured(path, 'rb')
    total_data_measure_end = time.time()
    total_data_measure_time = total_data_measure_end - total_data_measure_start
    data_measure_time = total_data_measure_time - load_time

    preprocess_start = time.time()
    input_data = (pd.read_csv(measured_data_bytes, names=column_names, na_values="?", sep=r'\s*,\s*', engine='python').loc[lambda df: df['race'].isin(['White', 'Black'])])
    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    sensitive_attribs = ['race', 'sex']
    Z = (input_data.loc[:, sensitive_attribs].assign(race=lambda df: (df['race'] == 'White').astype(int),sex=lambda df: (df['sex'] == 'Male').astype(int)))

    # targets; 1 when someone makes over 50k , otherwise 0
    y = (input_data['target'] == '>50K').astype(int)

    # features; note that the 'target' and sentive attribute columns are dropped
    X = (input_data.drop(columns=['target', 'race', 'sex', 'fnlwgt']).fillna('Unknown').pipe(pd.get_dummies, drop_first=True))

    X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(X, y, Z, test_size=0.5, stratify=y, random_state=0)
    X_train, y_train, Z_train = X_train.to_numpy(), y_train.to_numpy(), Z_train.to_numpy()
    X_test, y_test, Z_test = X_test.to_numpy(), y_test.to_numpy(), Z_test.to_numpy()

    train_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_train,dtype=np.float64)).type(torch.FloatTensor), torch.from_numpy(np.array(y_train,dtype=np.int64)).type(torch.LongTensor), torch.from_numpy(np.array(Z_train,dtype=np.int64)).type(torch.FloatTensor))
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_test,dtype=np.float64)).type(torch.FloatTensor), torch.from_numpy(np.array(y_test,dtype=np.int64)).type(torch.LongTensor), torch.from_numpy(np.array(Z_test,dtype=np.int64)).type(torch.FloatTensor))
    preprocess_end = time.time()
    preprocess_time = preprocess_end - preprocess_start

    return train_data, test_data, measured_data_bytes.hasher.digest(), data_measure_time, load_time, preprocess_time


def process_imdb(path = './data/IMDB_Dataset.csv'):

    total_data_measure_start = time.time()
    measured_data_bytes, _, load_time = open_measured(path, 'rb')
    total_data_measure_end = time.time()
    total_data_measure_time = total_data_measure_end - total_data_measure_start
    data_measure_time = total_data_measure_time - load_time

    preprocess_start = time.time()
    df = pd.read_csv(measured_data_bytes)
    df.head()

    X,y = df['review'].values,df['sentiment'].values
    x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)
    # print(f'shape of train data is {x_train.shape}')
    # print(f'shape of test data is {x_test.shape}')

    # dd = pd.Series(y_train).value_counts()
    # sns.barplot(x=np.array(['negative','positive']),y=dd.values)
    # plt.show()

    def preprocess_string(s):
        # Remove all non-word characters (everything except numbers and letters)
        s = re.sub(r"[^\w\s]", '', s)
        # Replace all runs of whitespaces with no space
        s = re.sub(r"\s+", '', s)
        # replace digits with no space
        s = re.sub(r"\d", '', s)

        return s

    def tockenize(x_train,y_train,x_val,y_val):
        word_list = []

        stop_words = set(stopwords.words('english')) 
        for sent in x_train:
            for word in sent.lower().split():
                word = preprocess_string(word)
                if word not in stop_words and word != '':
                    word_list.append(word)
    
        corpus = Counter(word_list)
        # sorting on the basis of most common words
        corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
        # creating a dict
        onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}
        
        # tockenize
        final_list_train,final_list_test = [],[]
        for sent in x_train:
                final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                        if preprocess_string(word) in onehot_dict.keys()])
        for sent in x_val:
                final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                        if preprocess_string(word) in onehot_dict.keys()])
                
        encoded_train = [1 if label =='positive' else 0 for label in y_train]  
        encoded_test = [1 if label =='positive' else 0 for label in y_val] 
        return np.array(final_list_train, dtype='object'), np.array(encoded_train),np.array(final_list_test, dtype='object'), np.array(encoded_test),onehot_dict

    x_train,y_train,x_test,y_test,vocab = tockenize(x_train,y_train,x_test,y_test)

    def padding_(sentences, seq_len):
        features = np.zeros((len(sentences), seq_len),dtype=int)
        for ii, review in enumerate(sentences):
            if len(review) != 0:
                features[ii, -len(review):] = np.array(review, dtype='object')[:seq_len]
        return features

    x_train_pad = padding_(x_train,500)
    x_test_pad = padding_(x_test,500)

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))
    # vocab_size = len(vocab) + 1
    # print(vocab_size)

    preprocess_end = time.time()
    preprocess_time = preprocess_end - preprocess_start

    return train_data, test_data, measured_data_bytes.hasher.digest(), data_measure_time, load_time, preprocess_time


def load_dataset_from_disk(path):
    dataset_dict = {}
    hashes = {}
    total_load_time = 0

    has_subdirs = any(os.path.isdir(os.path.join(path, split)) for split in ['train', 'test'])
    splits_to_process = ['train', 'test'] if has_subdirs else ['']

    print("Loading:", path)
    for split in splits_to_process:
        split_path = os.path.join(path, split) if split else path
        state_path = os.path.join(split_path, "state.json")

        if not os.path.exists(state_path):
            continue

        with open(state_path, "r") as f:
            state = json.load(f)
        arrow_files = [entry["filename"] for entry in state.get("_data_files", [])]

        split_datasets = []
        split_hashes = {}

        for fname in tqdm(arrow_files, desc=f"Loading {split or 'default'} split"):
            full_path = os.path.join(split_path, fname)
            buffer_ds, chunk_hash, load_time = open_measured(full_path, "rb")
            ds = HFDataset.from_buffer(buffer_ds)
            total_load_time += load_time
            split_datasets.append(ds)
            split_hashes[fname] = chunk_hash

        if split_datasets:
            dataset_dict[split or 'train'] = concatenate_datasets(split_datasets)
            hashes[split or 'train'] = split_hashes

    return DatasetDict(dataset_dict), hashes, total_load_time


class CustomTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        # train_dataset = self.train_dataset
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=SequentialSampler(self.train_dataset),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=False,
        )
