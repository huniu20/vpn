from transformers import BertModel,BertConfig,BertTokenizer, InputExample, InputFeatures
import torch
import random
from typing import List, Dict
from data_process import *
# import data_process
import json
from functools import lru_cache
from collections import defaultdict


class WEIBODataset:
    """
    微博数据集
    """
    train_file = "/home/liuhq/hun/my_work/datasets/weibo/simple_weiboNER_2nd_conll.train"
    test_file = "/home/liuhq/hun/my_work/datasets/weibo/simple_weiboNER_2nd_conll.test"
    label_map = "/home/liuhq/hun/my_work/datasets/weibo/label_map.json"
    tag_sep = " "
    char_sep = "\n"
    sentence_sep = "\n\n"
    backbone = "bert-base-chinese"
    num_tags = 9

class ONTONOTE4Dataset:
    """
    OntoNote4.0数据集
    """
    train_file = "/home/liuhq/hun/my_work/datasets/weibo/simple_weiboNER_2nd_conll.train"
    test_file = "/home/liuhq/hun/my_work/datasets/weibo/simple_weiboNER_2nd_conll.test"
    label_map = "/home/liuhq/hun/my_work/datasets/weibo/label_map.json"
    tag_sep = " "
    char_sep = "\n"
    sentence_sep = "\n\n"

class Param():
    def __init__(self) -> None:
        self.dataset_name = "weibo"
        self.top_k = 5
        self.batch_size = 8
        
        
        
    @property
    def entity_ids_to_word_ids(self):
        return {0: ([8024, 4638, 511, 120, 2769], [0.34385256248422114, 0.22923504165614744, 0.15501136076748295, 0.14642766978035848, 0.12547336531178996]),
                1: ([2207, 1957, 4511, 5439, 6627], [0.3355048859934853, 0.23452768729641693, 0.17263843648208468, 0.13680781758957655, 0.12052117263843648]),
                2: ([782, 1351, 2094, 3173, 3696], [0.35311572700296734, 0.22255192878338279, 0.20178041543026706, 0.11275964391691394, 0.10979228486646884]),
                5: ([2157, 2191, 2162, 7270, 100], [0.25, 0.1875, 0.1875, 0.1875, 0.1875]),
                6: ([2255, 1736, 6125, 2270, 3441], [0.23333333333333334, 0.23333333333333334, 0.23333333333333334, 0.16666666666666666, 0.13333333333333333]),
                3: ([5401, 677, 704, 7506, 1266], [0.2222222222222222, 0.2222222222222222, 0.20634920634920634, 0.1746031746031746, 0.1746031746031746]),
                4: ([1744, 3862, 776, 2336, 2356], [0.4639175257731959, 0.18556701030927836, 0.15463917525773196, 0.1134020618556701, 0.08247422680412371]),
                7: ([7032, 1290, 4510, 2128, 736], [0.25925925925925924, 0.2222222222222222, 0.18518518518518517, 0.18518518518518517, 0.14814814814814814]),
                8: ([3857, 2110, 2421, 5381, 7368], [0.4647887323943662, 0.23943661971830985, 0.09859154929577464, 0.09859154929577464, 0.09859154929577464])}
        
    @property
    def dataset_cls(self):
        return globals()[self.dataset_name.upper() + "Dataset"]
    
    @property
    def train_file(self):
        return self.dataset_cls.train_file
    
    @property
    def test_file(self):
        return self.dataset_cls.test_file
    
    @property
    def num_tags(self):
        return self.dataset_cls.num_tags
    
    @property
    def label_map(self):
        with open(self.dataset_cls.label_map, "r", encoding="utf-8") as f:
            res = json.load(f)
        return res

    @property
    def tokenizer(self):
        return BertTokenizer.from_pretrained(self.dataset_cls.backbone)

    @property
    def entity_to_words(self):
        entity_words_dict = {}
        examples = load_dataset(self.dataset_cls.train_file, self)
        dic = self.get_str_list_label_specificed(examples)
        for key, val in dic.items():
            entity_words_dict[key] = [[word_freq[0], word_freq[1]] for word_freq in self.compute_freq(dic[key], self.top_k * 4)]
        return entity_words_dict
    
    def get_str_list_label_specificed(self, examples):
        """生成某类实体及该实体对应的所有token

        Args:
            examples (_type_): _description_
        """
        labels = []
        chars = []
        for example in examples:
            labels.extend(example.labels)
            chars.extend(example.chars)
        
        unique_labels = list(set(labels))
        # print(unique_labels)
        str_list_label_specificed = defaultdict(list)
        for idx,label in enumerate(labels):
            for l in unique_labels:
                if label == l:
                    str_list_label_specificed[l].append(chars[idx])
        self.num_over_type = []
        for key in str_list_label_specificed.keys():
            print(f"{key}类有{len(str_list_label_specificed[key])}个")
            self.num_over_type.append((key, len(str_list_label_specificed[key])))
        return str_list_label_specificed
    
    @property
    def entity_id_to_words_id(self) -> Dict[int, int]:
        entity_to_words = self.filter_repetition_label_words()
        res = {}
        for entity, words in entity_to_words.items():
            entity_id = self.label_map[entity]
            word_ids = []
            word_freq = []
            word_top_k = []
            sum_entity_freq = 0
            for word in words:
                try:
                    word_ids.append(self.tokenizer._convert_token_to_id(word[0]))
                    word_freq.append(word[1])
                    sum_entity_freq += word[1]
                    word_top_k.append(word)
                    if len(word_ids) >= self.top_k:break
                except:
                    if len(word_ids) >= self.top_k:break
            word_freq = [f / sum_entity_freq for f in word_freq]
            res[entity_id] = (word_ids, word_freq)
        return res
        # print(entity_to_words)
    
    def compute_freq(self, chars, K=None):
        word_freq = {}
        for word in chars:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1
        sorted_word_freq = sorted(word_freq.items(), key=lambda x:x[1], reverse=True)
        return sorted_word_freq[:K]
    
    def filter_repetition_label_words(self):
        entity_words_dict = self.entity_to_words
        num_over_type = self.num_over_type
        print(num_over_type)
        num_over_type.sort(key=lambda x:x[1])
        entity_words_dict_no_freq = {}
        entity_words_dict_filtered = defaultdict(list)
        for key, val in entity_words_dict.items():
            # 仅含字，不含对应频次的字典
            entity_words_dict_no_freq[key] = [item[0] for item in val]
        max_word_freq = []
        all_char = []
        for i in range(len(num_over_type)):
            key, val = num_over_type[i][0], entity_words_dict[key]
            # print(key)
            for _val in val:
                if _val[0] not in all_char:
                    all_char.append(_val[0])
                    max_word_freq.append(_val)
                else:
                    for item in max_word_freq:
                        if _val[0] == item[0] and _val[1] >= item[1]:
                            max_word_freq.remove(item)
                            max_word_freq.append(_val)
        for key, val in entity_words_dict.items():
            entity_words_dict_filtered[key] = [v for v in val if v in max_word_freq][:self.top_k]
            if len(entity_words_dict_filtered[key]) < self.top_k:
                entity_words_dict_filtered[key] = entity_words_dict[key][:self.top_k]
        return entity_words_dict_filtered
    
if __name__ == "__main__":
    p = Param()
    print(p.test_file)
    print(p.label_map)
    # data = load_dataset(path=p.test_file)
    print(p.entity_id_to_words_id)
    print(p.tokenizer.get_vocab()["[PAD]"])
    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # print(len(tokenizer.get_vocab()))
    # print(tokenizer._convert_token_to_id("女"))