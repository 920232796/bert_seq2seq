import torch 
import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_ner")
import codecs
from tqdm import tqdm
import torch.nn as nn 
from torch.optim import Adam
import numpy as np
import os
import json
import time
import unicodedata
import bert_seq2seq
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert, load_model_params, load_recent_model

target = ["O", "B-DRUG", "B-DRUG_INGREDIENT", "B-DISEASE", "B-SYMPTOM", "B-SYNDROME", "B-DISEASE_GROUP", 
        "B-FOOD", "B-FOOD_GROUP", "B-PERSON_GROUP", "B-DRUG_GROUP", "B-DRUG_DOSAGE", "B-DRUG_TASTE",
         "B-DRUG_EFFICACY", "I-DRUG", "I-DRUG_INGREDIENT", "I-DISEASE", "I-SYMPTOM", "I-SYNDROME", "I-DISEASE_GROUP", 
        "I-FOOD", "I-FOOD_GROUP", "I-PERSON_GROUP", "I-DRUG_GROUP", "I-DRUG_DOSAGE", "I-DRUG_TASTE",
         "I-DRUG_EFFICACY"]

# target = ["O", "DRUG", "DRUG_INGREDIENT", "DISEASE", "SYMPTOM", "SYNDROME", "DISEASE_GROUP", 
#         "FOOD", "FOOD_GROUP", "PERSON_GROUP", "DRUG_GROUP", "DRUG_DOSAGE", "DRUG_TASTE",
#          "DRUG_EFFICACY"]

labels2id = {k: v for v, k in enumerate(target)}

vocab_path = "./roberta_wwm_vocab.txt" # roberta模型字典的位置   
model_name = "roberta" # 选择模型名字
model_path = "./roberta_wwm_pytorch_model.bin" # roberta模型位置
recent_model_path = "" # 用于把已经训练好的模型继续训练
model_save_path = "./bert_ner_model_crf.bin"
batch_size = 8
lr = 1e-5
crf_lr = 1e-2 ##  crf层学习率为0.01
# 加载字典
word2idx = load_chinese_base_vocab(vocab_path, simplfied=True)


def from_ann2dic(w_path):

    for i in range(1000):
        print(i)
        r_ann_path = "./corpus/医学NER/train/" + str(i) + ".ann"
        r_txt_path = "./corpus/医学NER/train/" + str(i) + ".txt"
        q_dic = {}
        print("开始读取文件:%s" % r_ann_path)
        with codecs.open(r_ann_path, "r", encoding="utf-8") as f:
            line = f.readline()
            line = line.strip("\n\r")
            while line != "":
                line_arr = line.split()
                # print(line_arr)
                cls = line_arr[1]
                start_index = int(line_arr[2])
                end_index = int(line_arr[3])
                length = end_index - start_index
                for r in range(length):
                    if r == 0:
                        q_dic[start_index] = ("B-%s" % cls)
                    else:
                        q_dic[start_index + r] = ("I-%s" % cls)
                line = f.readline()
                line = line.strip("\n\r")

        print("开始读取文件:%s" % r_txt_path)
        with codecs.open(r_txt_path, "r", encoding="utf-8") as f:
            content_str = f.read()
            content_str = content_str.replace("、", "，")
        print("开始写入文本%s" % w_path)
        with codecs.open(w_path, encoding="utf-8", mode="a+") as w:
            for i, char in enumerate(content_str):
                if char == " " or char == "" or char == "\n" or char == "\r" or char == "<" or char == ">" or char == "b" or char == "r" or char == "/" or unicodedata.category(char) == 'Zs' or char == "-":
                    continue
                
                else:
                    if i in q_dic:
                        tag = q_dic[i]
                    else:
                        tag = "O"  # 大写字母O
                    w.write('%s %s\n' % (char, tag))
            # w.write('%s\n' % "END O")
            

def load_data(path: str):
    """
    加载数据
    """
    src_data = []
    labels_data = []
    with open(path) as f :
        line = f.readline()
        line = line.strip("\n")
        temp_list = ""
        temp_label_list = [0]
        # index = 0
        while line != "":
            # index += 1
            # if index == 650:
            #     break
            ##开始一行一行读数据
            line_split = line.split(" ")
            # print(line_split)
            if line_split[0] == "。":
                # temp_list += (line_split[0])
                # temp_label_list.append(labels2id[line_split[1]])
                temp_label_list.append(0)
                src_data.append(temp_list)
                labels_data.append(temp_label_list)
                temp_list = ""
                temp_label_list = [0]
            else :
                temp_list += (line_split[0])
                temp_label_list.append(labels2id[line_split[1]])
                

            line = f.readline()
            line = line.strip("\n")
    print("原始数据大小为：" + str(len(src_data)))
    save_src_data = []
    save_label_data = []
    for src, label in zip(src_data, labels_data):
        if len(src) < 5:
            # print(src)
            continue
        save_src_data.append(src)
        save_label_data.append(label)
        # retain = 0
        # # print(label)
        # for l in label:
        #     if l != 0:
        #         retain = 1
        #         break
        # if retain == 1:
        #     save_src_data.append(src)
        #     save_label_data.append(label)

        # retain = 0
    print("清洗后数据大小为：" + str(len(save_src_data)))
    # print("删除全O的句子后数据大小为：" + str(len(save_src_data)))
    return save_src_data, save_label_data
  
## 自定义dataset
class NERDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, sents_src, sents_tgt) :
        ## 一般init函数是加载所有数据
        super(NERDataset, self).__init__()
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

        if len(token_ids) != len(tgt):
           
            print(len(token_ids))
            print(len(tgt))
            print(src)
            print(self.tokenizer.decode(token_ids))
            print(tgt)
            self.__getitem__(i + 1)

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
  
    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    
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

def ner_print(model, test_data, device="cpu"):
    model.eval()
    
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
                decode.append("O")
        flag = 0
        print(decode)
        res = {}
        for index, each_entity in enumerate(decode):
            if each_entity != "O":
                if flag != each_entity:
                    # print(index - 1)
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
        self.sents_src, self.sents_tgt = load_data("./res.txt")
        
        self.tokenier = Tokenizer(word2idx)
        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(word2idx, model_name=model_name, model_class="sequence_labeling_crf", target_size=len(target))
        ## 加载预训练的模型参数～
        load_model_params(self.bert_model, model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.to(self.device)
        # 声明需要优化的参数
        crf_params = list(map(id, self.bert_model.crf_layer.parameters())) ## 单独把crf层参数拿出来
        base_params = filter(lambda p: id(p) not in crf_params, self.bert_model.parameters())
        self.optimizer = torch.optim.Adam([
                                            {"params": base_params}, 
                                            {"params": self.bert_model.crf_layer.parameters(), "lr": crf_lr}], lr=lr, weight_decay=1e-5)
        # 声明自定义的数据加载器
        dataset = NERDataset(self.sents_src, self.sents_tgt)
        self.dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

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
            if step % 300 == 0:
                test_data = ["补气养血，调经止带，用于月经不调经期腹痛，非处方药物甲类，国家基本药物目录2012如果服用任何其他药品请告知医师或药师包括任何从药房超市或保健品商店购买的非处方药品。", 
                "月经过多孕妇忌服。黑褐色至黑色的小蜜丸味甜微苦。", 
                "红虎灌肠液50毫升装，安徽天洋药业清热解毒，化湿除带，祛瘀止痛，散结消癥，用于慢性盆腔炎所致小腹疼痛腰，骶酸痛带下量多或有发热。"]
                ner_print(self.bert_model, test_data, device=self.device)
                self.bert_model.train()

            token_ids = token_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                labels=target_ids,
                                                # use_layer_num=3
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
    
    # from_ann2dic("./res.txt")

    trainer = Trainer()
    train_epoches = 50
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)

    #  src_data, src_labels = load_data("./res.txt")
    

# if __name__ == "__main__":

#     ## from_ann2dic("./res.txt")

#     src_data, src_labels = load_data("./res.txt")

#     print(len(src_data))
#     print(len(src_labels))



    