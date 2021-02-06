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
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert

vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置
model_name = "roberta" # 选择模型名字
model_path = "./state_dict/roberta_wwm_pytorch_model.bin" # roberta模型位置
recent_model_path = "./bert_model_poem.bin" # 用于把已经训练好的模型继续训练
model_save_path = "./bert_model_poem.bin"
batch_size = 16
lr = 1e-5

word2idx, keep_tokens = load_chinese_base_vocab(vocab_path, simplfied=True)

def read_corpus(dir_path):
    """
    读原始数据
    """
    sents_src = []
    sents_tgt = []
    
    tokenizer = Tokenizer(word2idx)
    files= os.listdir(dir_path) #得到文件夹下的所有文件名称
    for file1 in files: #遍历文件夹
        if not os.path.isdir(file1): #判断是否是文件夹，不是文件夹才打开
            file_path = dir_path + "/" + file1
            print(file_path)
            if file_path[-3:] != "csv":
                continue
            df = pd.read_csv(file_path)
            # 先判断诗句的类型  再确定是否要构造数据
            for index, row in df.iterrows():
                if type(row[0]) is not str or type(row[3]) is not str:
                    continue
                if len(row[0].split(" ")) > 1:
                    # 说明题目里面存在空格，只要空格前面的数据
                    row[0] = row[0].split(" ")[0]

                if len(row[0]) > 10 or len(row[0]) < 1:
                    # 过滤掉题目长度过长和过短的诗句
                    continue
                
                encode_text = tokenizer.encode(row[3])[0]
                if word2idx["[UNK]"] in encode_text:
                  # 过滤unk字符
                  continue
                if len(row[3]) == 24 and (row[3][5] == "，" or row[3][5] == "。"):
                    # 五言绝句
                    sents_src.append(row[0] + "##" + "五言绝句")
                    sents_tgt.append(row[3])
                elif len(row[3]) == 32 and (row[3][7] == "，" or row[3][7] == "。"):
                    # 七言绝句
                    sents_src.append(row[0] + "##" + "七言绝句")
                    sents_tgt.append(row[3])
                elif len(row[3]) == 48 and (row[3][5] == "，" or row[3][5] == "。"):
                    # 五言律诗
                    sents_src.append(row[0] + "##" + "五言律诗")
                    sents_tgt.append(row[3])
                elif len(row[3]) == 64 and (row[3][7] == "，" or row[3][7] == "。"):
                    # 七言律诗
                    sents_src.append(row[0] + "##" + "七言律诗")
                    sents_tgt.append(row[3])

    # print("第一次诗句共: " + str(len(sents_src)) + "篇")
    return sents_src, sents_tgt

class BertDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, sents_src, sents_tgt) :
        ## 一般init函数是加载所有数据
        super(BertDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        
        self.idx2word = {k: v for v, k in word2idx.items()}
        self.tokenizer = Tokenizer(word2idx)

    def __getitem__(self, i):
        ## 得到单个数据
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

class PoemTrainer:
    def __init__(self):
        # 加载数据
        data_dir = "./corpus/Poetry"
        self.sents_src, self.sents_tgt = read_corpus(data_dir)
        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(word2idx, model_name=model_name)
        ## 加载预训练的模型参数～
        self.bert_model.load_pretrain_params(model_path, keep_tokens=keep_tokens)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.set_device(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = BertDataset(self.sents_src, self.sents_tgt)
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
            
            if step % 3000 == 0:
                self.bert_model.eval()
                test_data = ["北国风光##五言绝句", "题西林壁##七言绝句", "长安早春##五言律诗"]
                for text in test_data:
                    print(self.bert_model.generate(text, beam_size=3, is_poem=True))
                self.bert_model.train()

            token_ids = token_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                token_type_ids,
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

    trainer = PoemTrainer()
    train_epoches = 50
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)
