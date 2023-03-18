from transformers import BertModel,BertConfig,BertTokenizer, InputFeatures
import torch
import random
from typing import List
from transformers import DataCollatorForTokenClassification
# sfrom param import Param
import copy, json,tqdm,multiprocessing
from torch.utils.data import Dataset
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

class InputExample:
    def __init__(self, chars=None, labels=None, tags=None, ids=None) -> None:
        self.chars = chars # List[str]
        self.labels = labels # List[str]
        self.tags = tags # List[int]
        self.ids = ids # List[int]

    def __repr__(self) -> str:
        return str(self.__dict__) ### str(self.chars) + "\n" + str(self.labels) + "\n" + str(self.tags) + "\n" + str(self.ids)

def load_dataset(data_file, param):
    """加载数据

    Args:
        param (Param): _description_
    """
    examples = []
    if param.dataset_name == "msra":
        import json
        with open(data_file, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f.read().splitlines()]
        for line in lines:
            chars = list(line["text"])
            labels = ["O" for _ in range(len(chars))]
            if line["entity_list"] is not None:
                for entity in line["entity_list"]:
                    labels[entity["entity_index"]["begin"]] = "B-" + entity["entity_type"]
                    for i in range(entity["entity_index"]["begin"] + 1, entity["entity_index"]["end"]):
                        labels[i] = "I-" + entity["entity_type"]
            # print(chars, labels)
            # print(param.tokenizer.convert_tokens_to_ids(chars))
            # sdfa
            examples.append(InputExample(chars=chars, labels=labels))
    elif param.dataset_name == "ontonote4":
        with open(data_file, "r", encoding="utf-8") as f:
            data = f.read().strip()
            sentences = data.split(param.dataset_cls.sentence_sep)
            for sentence in sentences:
                chars = sentence.split(param.dataset_cls.char_sep)
                c = []
                l = []
                for char in chars:
                    if char == None:
                        continue
                    char = char.split(param.dataset_cls.tag_sep)
                    c.append(char[0])
                    if "M-" in char[1] or "E-" in char[1]:
                        cur_l = "I-" + char[1].split("-")[1]
                        l.append(cur_l)
                    elif "S-" in char[1]:
                        cur_l = "B-" + char[1].split("-")[1]
                        l.append(cur_l)
                    else:
                        l.append(char[1])

                examples.append(InputExample(chars=c, labels=l))
    else:
        with open(data_file, "r", encoding="utf-8") as f:
            data = f.read().strip()
            sentences = data.split(param.dataset_cls.sentence_sep)
            for sentence in sentences:
                chars = sentence.split(param.dataset_cls.char_sep)
                c = []
                l = []
                for char in chars:
                    if char == None:
                        continue
                    char = char.split(param.dataset_cls.tag_sep)
                    c.append(char[0])
                    if "." in char[1]:
                        cur_l = char[1].split(".")[0]
                        l.append(cur_l)
                    else:
                        l.append(char[1])
                examples.append(InputExample(chars=c, labels=l))
            # print(examples)
    return examples

class NERDataset(Dataset):
    def __init__(self, input_ids, attention_masks, label_ids) -> None:
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.label_ids = label_ids

    def __getitem__(self, index):
        return (self.input_ids[index], self.attention_masks[index], self.label_ids[index])

    def __len__(self):
        return len(self.input_ids)


def tokenize_one_example(example,param):
    chars = example.chars
    labels = example.labels
    tokenizer = BertTokenizer.from_pretrained(param.dataset_cls.backbone)
    chars_tokenized = [tokenizer.tokenize(t) for t in chars]
    input_ids = []
    label_ids = []
    input_ids.append(tokenizer.vocab["[CLS]"])
    label_ids.append(param.label_map["O"])
    for idx,char_tokenized in enumerate(chars_tokenized):
        label = labels[idx]
        input_ids.extend([tokenizer.vocab[t] for t in char_tokenized])
        if label.startswith("O"):
            label_ids.extend([param.label_map[label]] * len(char_tokenized))
        elif label.startswith("I-"):
            label_ids.extend([param.label_map[label]] * len(char_tokenized))
        else:
            label_ids.append(param.label_map[label])
            label_ids.extend([param.label_map["I-" + label[2:]]] * (len(char_tokenized) - 1))
    assert len(input_ids) == len(label_ids)
    if len(input_ids) > (param.max_len - 1):
        input_ids, label_ids = input_ids[:510], label_ids[:510]
    input_ids.append(tokenizer.vocab["[SEP]"])
    label_ids.append(param.label_map["O"])
    attention_mask = [1] * len(label_ids)
    return input_ids, label_ids, attention_mask
    
def build_features(examples, param):
    if not isinstance(examples, list):
        examples = [examples]
    for example in examples:
        input_ids, label_ids, attention_mask = tokenize_one_example(example, param)
        # else:
        #     ids = []
        #     for char in example.chars:
        #         try:
        #             ids.append(param.vocab[char])
        #         except:
        #             ids.append(100)
        #     feature_map["input_ids"].append(ids)
        #     feature_map["label_ids"].append([param.label_map[e] for e in example.labels[:510]])
        #     feature_map["attention_mask"].append([1 for _ in range(len([param.label_map[e] for e in example.labels[:510]]))])

    return input_ids, label_ids, attention_mask

def collate_fn(batch_data, pad=0, label_pad=0):
    # B-LOC
    # B-LOC I-LOC
    # I-LOC
    # I-LOC I-LOC
    
    batch_input_ids, batch_label_ids, batch_attention_mask = list(zip(*batch_data))
    max_len = max([len(seq) for seq in batch_input_ids])
    # print(batch_input_ids)
    batch_input_ids = [seq + [pad]*(max_len-len(seq))  for seq in batch_input_ids]
    batch_label_ids = [seq + [label_pad]*(max_len-len(seq)) for seq in batch_label_ids]
    batch_attention_mask = [seq + [0]*(max_len-len(seq))  for seq in batch_attention_mask]
    batch_input_ids = torch.LongTensor(batch_input_ids)
    batch_label_ids = torch.LongTensor(batch_label_ids)
    batch_attention_mask = torch.FloatTensor(batch_attention_mask)
    return batch_input_ids, batch_label_ids, batch_attention_mask

if __name__ == "__main__":
    from param import Param
    from functools import partial
    p = Param()
    build_features_new = partial(build_features, param=p)
    print(p.test_file)
    x = load_dataset(p.test_file, p)
    print(type(x))
    # print(type(x))
    # print(x[:3])
    # print(len(x))
    # feature_map = build_features(x, p)
    # sdf
    # spool = ProcessPoolExecutor()
    with ProcessPoolExecutor(100) as pool:
        output = list(pool.map(build_features_new,x))
    # proc = multiprocessing.Process(target=build_features_new,args=[x)
    # feature_map = proc.run()
    print(len(output))
    input, label, atten = list(zip(*output))
    # print(output[1], type(output[1]))
    train_data = NERDataset(input, atten,label)
    print(train_data[1])
    print(type(train_data), type(train_data[1]))
    print(len(train_data))