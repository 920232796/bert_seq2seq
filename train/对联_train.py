## 情感分析例子
import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/large_bert")
import torch 
from tqdm import tqdm
import torch.nn as nn 
from torch.optim import Adam
import pandas as pd
import numpy as np
import os
import json
from config import sentiment_batch_size, sentiment_lr, duilian_corpus_dir, bert_chinese_model_path
from model.seq2seq_model import Seq2SeqModel
from model.bert_model import BertConfig
import time
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer, load_chinese_base_vocab

def read_corpus(data_path):
    """
    读原始数据
    """
    
    sents_src = []
    sents_tgt = []
    in_path = data_path + "/in.txt"
    out_path = data_path + "/out.txt"
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            sents_src.append(line.strip())
    with open(out_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            sents_tgt.append(line.strip())

    return sents_src, sents_tgt

## 自定义dataset
class DreamDataset(Dataset):
    """
    针对周公姐解梦数据集，定义一个相关的取数据的方式
    """
    def __init__(self) :
        ## 一般init函数是加载所有数据
        super(DreamDataset, self).__init__()
        # 读原始数据
        self.sents_src, self.sents_tgt = read_corpus(duilian_corpus_dir)
        self.word2idx = load_chinese_base_vocab()
        self.idx2word = {k: v for v, k in self.word2idx.items()}
        self.tokenizer = Tokenizer(self.word2idx)
        # print(self.sents_src[:3])

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        
        token_ids, token_type_ids = self.tokenizer.encode(src, tgt)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
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
        注意 token type id 右侧pad是添加1而不是0，1表示属于句子B
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)
   
    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded


class DreamTrainer:
    def __init__(self):
        # 加载情感分析数据
        self.pretrain_model_path = bert_chinese_model_path
        self.batch_size = sentiment_batch_size
        self.lr = sentiment_lr
        # 加载字典
        self.word2idx = load_chinese_base_vocab()
        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 定义模型超参数
        bertconfig = BertConfig(vocab_size=len(self.word2idx))
        # 初始化BERT模型
        self.bert_model = Seq2SeqModel(config=bertconfig)
        ## 加载预训练的模型～
        self.load_model(self.bert_model, self.pretrain_model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.to(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.init_optimizer(lr=self.lr)
        # 声明自定义的数据加载器
        dataset = DreamDataset()
        self.dataloader =  DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        
    def init_optimizer(self, lr):
        # 用指定的学习率初始化优化器
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)

    def load_model(self, model, pretrain_model_path):
        
        checkpoint = torch.load(pretrain_model_path)
        # 模型刚开始训练的时候, 需要载入预训练的BERT
        
        checkpoint = {k[5:]: v for k, v in checkpoint.items()
                                            if k[:4] == "bert" and "pooler" not in k}
        model.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        print("{} loaded!".format(pretrain_model_path))

    def train(self, epoch):
        # 一个epoch的训练
        self.bert_model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time() ## 得到当前时间
        ste = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader,position=0, leave=True):
            step += 1
            if step % 500 == 0:
                self.bert_model.eval()
                test_data = ["花海总涵功德水", "广汉飞霞诗似玉", "执政为民，德行天下"]
                for text in test_data:
                    print(self.bert_model.generate(text, beam_size=3,device=self.device))
                self.bert_model.train()

            token_ids = token_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                token_type_ids,
                                                labels=target_ids,
                                                device=self.device
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
        print(f"epoch is {epoch}. loss is {loss:06}. spend time is {spend_time}")
        # 保存模型
        self.bert_model.eval()
        test_data = ["花海总涵功德水", "广汉飞霞诗似玉", "执政为民，德行天下"]
        for text in test_data:
            print(self.bert_model.generate(text, beam_size=3,device=self.device))
        self.bert_model.train()
        self.save_state_dict(self.bert_model, epoch)

    def save_state_dict(self, model, epoch, file_path="bert_duilian.model"):
        """存储当前模型参数"""
        save_path = "./" + file_path + ".epoch.{}".format(str(epoch))
        torch.save(model.state_dict(), save_path)
        print("{} saved!".format(save_path))

if __name__ == '__main__':

    # word2idx = load_chinese_base_vocab()
    # tokenier = Tokenizer(word2idx)

    trainer = DreamTrainer()
    train_epoches = 10
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)

    ## 测试一下read corpus
    # sents_src, sents_tgt = read_corpus(duilian_corpus_dir)
    # print(sents_src[:5])
    # print(sents_tgt[:5])
    # print(tokenier.encode(sents_src[0]))

    # # 测试一下自定义数据集
    # dataset = DreamDataset()
    # word2idx = load_chinese_base_vocab()
    # tokenier = Tokenizer(word2idx)
    # dataloader =  DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    # for token_ids, token_type_ids, target_ids in dataloader:
    #     print(token_ids.shape)
    #     print(tokenier.decode(token_ids[0].tolist()))
    #     print(tokenier.decode(token_ids[1].tolist()))
    #     print(token_type_ids)
    #     print(target_ids.shape)
    #     print(tokenier.decode(target_ids[0].tolist()))
    #     print(tokenier.decode(target_ids[1].tolist()))
    #     break

    