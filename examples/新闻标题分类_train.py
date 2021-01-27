## 文本分类的例子
import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")
import torch 
from tqdm import tqdm
import torch.nn as nn 
from torch.optim import Adam
import pandas as pd
import numpy as np
import os
import json
import time
import bert_seq2seq
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert

target = ["财经", "彩票", "房产", "股票", "家居", "教育", "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"]

vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置
model_name = "roberta" # 选择模型名字
model_path = "./state_dict/roberta_wwm_pytorch_model.bin" # roberta模型位置
recent_model_path = "" # 用于把已经训练好的模型继续训练
model_save_path = "./bert_multi_classify_model.bin"
batch_size = 16
lr = 1e-5
# 加载字典
word2idx = load_chinese_base_vocab(vocab_path)

def read_corpus(data_path):
    """
    读原始数据
    """
    sents_src = []
    sents_tgt = []
    
    with open(data_path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.split("\t")
        sents_tgt.append(int(line[0]))
        sents_src.append(line[2])
    return sents_src, sents_tgt

## 自定义dataset
class NLUDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, sents_src, sents_tgt) :
        ## 一般init函数是加载所有数据
        super(NLUDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
    
        self.idx2word = {k: v for v, k in word2idx.items()}
        self.tokenizer = Tokenizer(word2idx)

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        token_ids, token_type_ids = self.tokenizer.encode(src)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
            "target_id": tgt
        }
        return output

    def __len__(self):
        return len(self.sents_src)
    
def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    target_ids = [data["target_id"] for data in batch]
    target_ids = torch.tensor(target_ids, dtype=torch.long)

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    # target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids

class Trainer:
    def __init__(self):
        # 加载数据
        data_path = "./corpus/新闻标题文本分类/Train.txt"
        self.sents_src, self.sents_tgt = read_corpus(data_path)
        self.tokenier = Tokenizer(word2idx)
        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(word2idx, model_name=model_name, model_class="cls", target_size=len(target))
        ## 加载预训练的模型参数～
        self.bert_model.load_pretrain_params(model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.to(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = NLUDataset(self.sents_src, self.sents_tgt)
        self.dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def train(self, epoch):
        # 一个epoch的训练
        self.bert_model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)
    
    def save(self, save_path):
        """
        保存模型
        """
        self.bert_model.save_all_params(save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time() ## 得到当前时间
        step = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader,position=0, leave=True):
            step += 1
            if step % 2000 == 0:
                self.bert_model.eval()
                test_data = ["编剧梁馨月讨稿酬六六何念助阵 公司称协商解决", "西班牙BBVA第三季度净利降至15.7亿美元", "基金巨亏30亿 欲打开云天系跌停自救"]
                for text in test_data:
                    text, text_ids = self.tokenier.encode(text)
                    text = torch.tensor(text, device=self.device).view(1, -1)
                    print(target[torch.argmax(self.bert_model(text)).item()])
                self.bert_model.train()

            token_ids = token_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                labels=target_ids,                                
                                                )
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()
        
        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch)+". loss is " + str(total_loss) + ". spend time is "+ str(spend_time))
        # 保存模型
        self.save(model_save_path)

if __name__ == '__main__':
    
    trainer = Trainer()
    train_epoches = 10
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)