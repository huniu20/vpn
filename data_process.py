from transformers import BertModel,BertConfig,BertTokenizer, InputFeatures
import torch
import random
from typing import List
from transformers import DataCollatorForTokenClassification
# from param import Param
import copy, json,tqdm,multiprocessing
from torch.utils.data import Dataset
from collections import defaultdict

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


def build_features(examples, param):
    feature_map = defaultdict(list)
    # print("???")
    for example in tqdm.tqdm(examples, desc="building examples.... "):
        # print(example)
        if len(example.chars) <= 510:
            feature_map["input_ids"].append(param.tokenizer.convert_tokens_to_ids(example.chars))
            feature_map["label_ids"].append([param.label_map[e] for e in example.labels])
            feature_map["attention_mask"].append([1 for _ in range(len([param.label_map[e] for e in example.labels]))])
        else:
            feature_map["input_ids"].append(param.tokenizer.convert_tokens_to_ids(example.chars[:510]))
            feature_map["label_ids"].append([param.label_map[e] for e in example.labels[:510]])
            feature_map["attention_mask"].append([1 for _ in range(len([param.label_map[e] for e in example.labels[:510]]))])
    return NERDataset(feature_map["input_ids"], feature_map["label_ids"], feature_map["attention_mask"])

def collate_fn(batch_data, pad=0, cls=101, sep=102):
    batch_input_ids, batch_label_ids, batch_attention_mask = list(zip(*batch_data)) # 解包
    max_len = max([len(seq) for seq in batch_input_ids])
    # print(batch_input_ids)
    batch_input_ids = [[cls] + seq + [pad]*(max_len-len(seq)) + [sep] for seq in batch_input_ids]
    batch_label_ids = [[-100] + seq + [-100]*(max_len-len(seq)) + [-100] for seq in batch_label_ids]
    batch_attention_mask = [[0] + seq + [0]*(max_len-len(seq)) + [0] for seq in batch_attention_mask]
    batch_input_ids = torch.LongTensor(batch_input_ids)
    batch_label_ids = torch.LongTensor(batch_label_ids)
    batch_attention_mask = torch.FloatTensor(batch_attention_mask)
    return batch_input_ids, batch_attention_mask, batch_label_ids

if __name__ == "__main__":
    from param import Param
    p = Param()
    print(p.test_file)
    x = load_dataset(p.test_file, p)
    # print(type(x))
    # print(x[:3])
    # print(len(x))
    feature_map = build_features(x, p)
    # proc = multiprocessing.Process(target=build_features,args=[x,p])
    # feature_map = proc.run()
    print(feature_map)