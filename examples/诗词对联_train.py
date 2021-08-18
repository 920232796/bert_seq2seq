## 自动写诗的例子
import torch
import pandas as pd
import os
import time
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert
import opencc 

data_dir = "./Poetry_ci_duilian"

vocab_path = "./roberta_wwm_vocab.txt" # roberta模型字典的位置
model_name = "roberta" # 选择模型名字
model_path = "./roberta_wwm_pytorch_model.bin" # roberta模型位置
recent_model_path = "./bert_model_poem_ci_duilian.bin" # 用于把已经训练好的模型继续训练
model_save_path = "./bert_model_poem_ci_duilian.bin"
batch_size = 8
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

    print("第一个诗句数据集共: " + str(len(sents_src)) + "篇")
    return sents_src, sents_tgt

def read_corpus_2(dir_path):
    """读取最近的一个数据集 唐诗和宋诗 """
    sents_src = []
    sents_tgt = []
    tokenizer = Tokenizer(word2idx)
    files= os.listdir(dir_path) #得到文件夹下的所有文件名称
    
    for file1 in files: #遍历文件夹
       
        if not os.path.isdir(file1): #判断是否是文件夹，不是文件夹才打开
            file_path = dir_path + "/" + file1
            print(file_path)
            # data = json.load(file_path)
            with open(file_path) as f :
                poem_list = eval(f.read())
            
            for each_poem in poem_list:
                string_list = each_poem["paragraphs"]
                poem = ""
                for each_s in string_list:
                    poem += each_s

                cc = opencc.OpenCC('t2s')
                poem = cc.convert(poem)

                encode_text = tokenizer.encode(poem)[0]
                if word2idx["[UNK]"] in encode_text:
                    # 过滤unk字符
                    continue
                title = cc.convert(each_poem["title"])

                if len(title) > 10 or len(title) < 1:
                    # 过滤掉题目长度过长和过短的诗句
                    continue

                if len(poem) == 24 and (poem[5] == "，" or poem[5] == "。"):
                    # 五言绝句
                    sents_src.append(title+ "##" + "五言绝句")
                    sents_tgt.append(poem)
                elif len(poem) == 32 and (poem[7] == "，" or poem[7] == "。"):
                    # 七言绝句
                    sents_src.append(title + "##" + "七言绝句")
                    sents_tgt.append(poem)
                elif len(poem) == 48 and (poem[5] == "，" or poem[5] == "。"):
                    # 五言律诗
                    sents_src.append(title + "##" + "五言律诗")
                    sents_tgt.append(poem)
                elif len(poem) == 64 and (poem[7] == "，" or poem[7] == "。"):
                    # 七言律诗
                    sents_src.append(title + "##" + "七言律诗")
                    sents_tgt.append(poem)

    print("第二个诗句数据集共:" + str(len(sents_src)) + "篇")
    return sents_src, sents_tgt


def read_corpus_ci(dir_path):
    """ 读取宋词数据集"""
    import json, sys
    import sqlite3
    from collections import OrderedDict
    tokenizer = Tokenizer(word2idx)

    c = sqlite3.connect(dir_path + '/ci.db')

    cursor = c.execute("SELECT name, long_desc, short_desc from ciauthor;")

    d = {"name": None, "description": None, "short_description": None}

    cursor = c.execute("SELECT rhythmic, author, content from ci;")

    d = {"rhythmic": None, "author": None, "paragraphs": None}

    # cis = []
    sents_src = []
    sents_tgt = []

    for row in cursor:
        ci = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
        ci["rhythmic"] = row[0]
        ci["author"] = row[1]
        ci["paragraphs"] = row[2].split('\n')
        string = ""
        for s in ci["paragraphs"]:
            if s == " >> " or s == "词牌介绍":
                continue
            string += s

        encode_text = tokenizer.encode(string)[0]
        if word2idx["[UNK]"] in encode_text:
            # 过滤unk字符
            continue
        sents_src.append(row[0] + "##词")
        sents_tgt.append(string)

        # cis.append(ci)

    # print(cis[:10])
    print("词共: " + str(len(sents_src)) + "篇")
    return sents_src, sents_tgt

def read_corpus_duilian(dir_path):
    """读取对联数据集 """
    sents_src = []
    sents_tgt = []
    in_path = dir_path + "/in.txt"
    out_path = dir_path + "/out.txt"
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            sents_src.append(line.strip() + "##对联")
    with open(out_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            sents_tgt.append(line.strip())

    print("对联共: " + str(len(sents_src)) + "篇")
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
        
        self.sents_src, self.sents_tgt = read_corpus(data_dir + "/Poetry1")
        sents_src2, sents_tgt2 = read_corpus_2(data_dir + "/Poetry2")
        sents_src3, sents_tgt3 = read_corpus_ci(data_dir)
        sents_src4, sents_tgt4 = read_corpus_duilian(data_dir)
        self.sents_src.extend(sents_src2)
        self.sents_src.extend(sents_src3)
        self.sents_src.extend(sents_src4)

        self.sents_tgt.extend(sents_tgt2)
        self.sents_tgt.extend(sents_tgt3)
        self.sents_tgt.extend(sents_tgt4)

        ## 保存下加载的数据 下次容易加载
        # torch.save(self.sents_src, "./poem_ci_duilian.src")
        # torch.save(self.sents_tgt, "./poem_ci_duilian.tgt")

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
        report_loss = 0
        # for token_ids, token_type_ids, target_ids in tqdm(dataloader,position=0, leave=True):
        for token_ids, token_type_ids, target_ids in dataloader:
            step += 1
            if step % 3000 == 0:
                print("3000 step loss is :" + str(report_loss))
                report_loss = 0
                self.bert_model.eval()
                test_data = ["北国风光##五言绝句", "题西林壁##七言绝句", "一年四季行好运##对联", "浣溪沙##词"]
                for text in test_data:
                    # if text[-1] == "句" or text[-1] == "诗":
                    #     print(self.bert_model.generate(text, beam_size=3,device=self.device, is_poem=True))
                    # else :
                    print(self.bert_model.generate(text, beam_size=3))
                self.bert_model.train()

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
            report_loss += loss.item()
            if step % 8000 == 0:
                self.save(model_save_path)
        
        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch)+". loss is " + str(total_loss) + ". spend time is "+ str(spend_time))
        # 保存模型
        

if __name__ == '__main__':

    trainer = PoemTrainer()
    train_epoches = 200
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)

    ## 测试一下自定义数据集
    # data_dir = "./Poetry_ci_duilian"
    # vocab_path = "./roberta_wwm_vocab.txt" # roberta模型字典的位置
    # sents_src, sents_tgt = read_corpus(data_dir + "/Poetry1", vocab_path)
    # sents_src2, sents_tgt2 = read_corpus_2(data_dir + "/Poetry2", vocab_path)
    # sents_src3, sents_tgt3 = read_corpus_ci(data_dir, vocab_path)
    # sents_src4, sents_tgt4 = read_corpus_duilian(data_dir)
    # sents_src.extend(sents_src2)
    # sents_src.extend(sents_src3)
    # sents_src.extend(sents_src4)

    # sents_tgt.extend(sents_tgt2)
    # sents_tgt.extend(sents_tgt3)
    # sents_tgt.extend(sents_tgt4)
    
    # print(sents_src[:10])
    # print(sents_tgt[:10])
    # dataset = BertDataset(sents_src, sents_tgt, vocab_path)
    # word2idx = load_chinese_base_vocab(vocab_path, simplfied=True)
    # tokenier = Tokenizer(word2idx)
    # dataloader =  DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    # for token_ids, token_type_ids, target_ids in dataloader:
    #     print(token_ids.shape)
    #     print(tokenier.decode(token_ids[0].tolist()))
    #     print(tokenier.decode(token_ids[1].tolist()))
    #     print(tokenier.decode(token_ids[3].tolist()))
    #     print(tokenier.decode(token_ids[2].tolist()))

    #     print(token_type_ids)
    #     print(target_ids.shape)
    #     print(tokenier.decode(target_ids[0].tolist()))
    #     print(tokenier.decode(target_ids[1].tolist()))
    #     break


    