from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class RretrainDataset(Dataset):
    #init
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length #输入给GPU的最大长度
        #使用huggingface的惰性加载，避免一次读入大文件
        self.samples = load_dataset('json', data_files=data_path, split='train') 

    #__len__
    def __len__(self):
        return len(self.samples)

    #__getitem__
    # 拿到的是jsonl的每一行
    def __getitem__(self, index):
        sample = self.samples[index]
    # tokenizer把文本转化为input_ids
        sample = self.tokenizer(
            str(sample['text']), #假设jsonl里又一个“text”字段，包含文本内容
            add_special_token = False,
            max_length= self.max_length -2,#留出BOS和EOS的位置
            truncation = True,
        ).input_ids              #.input_ids自动取出来
    # 需要加上EOS,BOS以及PAD填充
        tokens=[self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens +[self.tokenizer.pad_token_ids]*(self.max_length-len(
            tokens))#填充到max_length以对齐
        input_ids = torch.Tensor(input_ids,dtype=torch.long) #转为Tensor
    #需要自行编写labels，防止PAD参与计算
        labels = input_ids.clone()
        labels = [labels == self.tokenizer.pad_token_ids] = -100 #将PAD位置的标签
        # 设为-100，表示忽略这部分的loss计算
    # 需要编写attention_mask,告诉模型哪些位置有效，哪些是PAD
        attention_mask = (input_ids != self.tokenizer.pad_token_ids).long()#非PAD
        # 位置为1，PAD位置为0
    # 输出的是input_ids,attention_mask,labels
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    

