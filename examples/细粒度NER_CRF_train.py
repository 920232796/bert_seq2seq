## 自动写诗的例子
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
from bert_seq2seq.utils import load_bert, load_model_params, load_recent_model
# 共11个标签
target = ["other", "address", "book", "company", "game", "government", "movie", "name", "organization", "position", "scene"]

def read_corpus(data_path):
    """
    读原始数据
    """
    sents_src = []
    sents_tgt = []

    with open(data_path) as f:
        lines = f.readlines()
    for line in lines:
        line_dict = eval(line)
        sents_src.append(line_dict["text"])
        target_list = [0] * (len(line_dict["text"]) + 2) # 算上了cls 和 sep符号
        label_dict = line_dict["label"]
        for k, v in label_dict.items():
            t = target.index(k)
            for kk, vv in v.items():
                for l in vv:
                    start = l[0]
                    end = l[1]
                    for i in range(start + 1, end + 2):
                        target_list[i] = t
        sents_tgt.append(target_list)

    return sents_src, sents_tgt

## 自定义dataset
class NERDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, sents_src, sents_tgt, vocab_path) :
        ## 一般init函数是加载所有数据
        super(NERDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        self.word2idx = load_chinese_base_vocab(vocab_path)
        self.idx2word = {k: v for v, k in self.word2idx.items()}
        self.tokenizer = Tokenizer(self.word2idx)

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
        注意 token type id 右侧pad是添加1而不是0，1表示属于句子B
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    target_ids = [data["target_id"] for data in batch]
  
    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    # target_ids_padded = token_ids_padded[:, 1:].contiguous()
    target_ids_padded = padding(target_ids, max_length)

    return token_ids_padded, token_type_ids_padded, target_ids_padded

def viterbi_decode(nodes, trans):
    """
    维特比算法 解码
    nodes: (seq_len, target_size)
    trans: (target_size, target_size)
    """
    scores = nodes[0]
    scores[1:] -= 100000 # 刚开始标签肯定是"O"
    target_size = nodes.shape[1]
    seq_len = nodes.shape[0]
    labels = torch.arange(0, target_size).view(1, -1)
    path = labels
    for l in range(1, seq_len):
        scores = scores.view(-1, 1)
        M = scores + trans + nodes[l].view(1, -1)
        scores, ids = M.max(0)
        path = torch.cat((path[:, ids], labels), dim=0)
        # print(scores)
    # print(scores)
    return path[:, scores.argmax()]

def ner_print(model, test_data, vocab_path, device="cpu"):
    model.eval()
    word2idx = load_chinese_base_vocab(vocab_path)
    tokenier = Tokenizer(word2idx)
    trans = model.state_dict()["crf_layer.trans"]
    for text in test_data:
        decode = []
        text_encode, text_ids = tokenier.encode(text)
        text_tensor = torch.tensor(text_encode, device=device).view(1, -1)
        out = model(text_tensor).squeeze(0) # 其实是nodes
        labels = viterbi_decode(out, trans)
        starting = False
        for l in labels:
            if l > 0:
                label = target[l.item()]
                decode.append(label)
            else :
                decode.append("other")
        flag = 0
        res = {}
        for index, each_entity in enumerate(decode):
            if each_entity != "other":
                if flag != each_entity:
                    cur_text = text[index - 1]
                    if each_entity in res.keys():
                        res[each_entity].append(cur_text)
                    else :
                        res[each_entity] = [cur_text]
                    flag = each_entity
                elif flag == each_entity:
                    res[each_entity][-1] += text[index - 1]
            else :
                flag = 0
        print(res)

class Trainer:
    def __init__(self):
        # 加载数据
        data_path = "./corpus/细粒度NER/train.json"
        self.vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置
        self.sents_src, self.sents_tgt = read_corpus(data_path)
        self.model_name = "roberta" # 选择模型名字
        self.model_path = "./state_dict/roberta_wwm_pytorch_model.bin" # roberta模型位置
        self.recent_model_path = "" # 用于把已经训练好的模型继续训练
        self.model_save_path = "./细粒度_bert_ner_model_crf.bin"
        self.batch_size = 8
        self.lr = 1e-5
        # 加载字典
        self.word2idx = load_chinese_base_vocab(self.vocab_path)
        self.tokenier = Tokenizer(self.word2idx)
        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(self.vocab_path, model_name=self.model_name, model_class="sequence_labeling_crf", target_size=len(target))
        ## 加载预训练的模型参数～
        load_model_params(self.bert_model, self.model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.to(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=self.lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = NERDataset(self.sents_src, self.sents_tgt, self.vocab_path)
        self.dataloader =  DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def train(self, epoch):
        # 一个epoch的训练
        self.bert_model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)
    
    def save(self, save_path):
        """
        保存模型
        """
        torch.save(self.bert_model.state_dict(), save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time() ## 得到当前时间
        step = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader,position=0, leave=True):
            # print(target_ids.shape)
            step += 1
            if step % 500 == 0:
                test_data = ["在广州经营小古董珠宝店的潘凝已经收藏了200多款泰迪熊，其中不少更是老牌泰迪熊厂商史蒂夫、赫曼。", "美联物业市场研究部总监张大伟介绍：从23日营业税详细政策出台北京市场已经明显出现变化"]
                ner_print(self.bert_model, test_data, self.vocab_path, device=self.device)
                self.bert_model.train()

            token_ids = token_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                labels=target_ids      
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
        self.save(self.model_save_path)

if __name__ == '__main__':
    
    trainer = Trainer()
    train_epoches = 50
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)


    # sents_src, tgt = read_corpus("./corpus/细粒度NER/train.json")
    # print(sents_src[:2])
    # print(tgt[:2])