import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")
import torch 
import torch.nn as nn
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert, load_model_params, load_recent_model
import numpy as np 
import time 

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append({
                'text': l['text'],
                'spo_list': [(spo['subject'], spo['predicate'], spo['object'])
                             for spo in l['spo_list']]
            })
    return D

predicate2id, id2predicate = {}, {}
with open('./corpus/三元组抽取/all_50_schemas') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

class ExtractDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, data, vocab_path) :
        ## 一般init函数是加载所有数据
        super(ExtractDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.data = data
        self.word2idx = load_chinese_base_vocab(vocab_path)
        self.idx2word = {k: v for v, k in self.word2idx.items()}
        self.tokenizer = Tokenizer(self.word2idx)

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        d = self.data[i]
        token_ids, segment_ids = self.tokenizer.encode(d["text"], max_length=128)
        spoes = {}
        for s, p, o in d['spo_list']:
            s = self.tokenizer.encode(s)[0][1:-1]
            p = predicate2id[p]
            o = self.tokenizer.encode(o)[0][1:-1]
            s_idx = search(s, token_ids)
            o_idx = search(o, token_ids)
            if s_idx != -1 and o_idx != -1:
                s = (s_idx, s_idx + len(s) - 1)
                o = (o_idx, o_idx + len(o) - 1, p)
                if s not in spoes:
                    spoes[s] = []
                spoes[s].append(o)
        if spoes:
            # subject标签
            subject_labels = np.zeros((len(token_ids), 2))
            for s in spoes:
                subject_labels[s[0], 0] = 1
                subject_labels[s[1], 1] = 1
            # 随机选一个subject
            start, end = np.array(list(spoes.keys())).T
            start = np.random.choice(start)
            end = np.random.choice(end[end >= start])
            subject_ids = (start, end)
            # 对应的object标签
            object_labels = np.zeros((len(token_ids), len(predicate2id), 2))
            for o in spoes.get(subject_ids, []):
                object_labels[o[0], o[2], 0] = 1
                object_labels[o[1], o[2], 1] = 1
        
            output = {
                "token_ids": token_ids,
                "token_type_ids": segment_ids,
                "subject_labels": subject_labels,
                "subject_ids": subject_ids,
                "object_labels": object_labels,
            }
            return output
        else: 
            return self.__getitem__(i + 1)

    def __len__(self):

        return len(self.data)
        
def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """
    def padding(inputs, max_length=None, padding=0):
        """Numpy函数，将序列padding到同一长度
        """
        if max_length is None:
            max_length = max([len(x) for x in inputs])

        pad_width = [(0, 0) for _ in np.shape(inputs[0])]
        outputs = []
        for x in inputs:
            x = x[:max_length]
            pad_width[0] = (0, max_length - len(x))
            x = np.pad(x, pad_width, 'constant', constant_values=padding)
            outputs.append(x)

        return np.array(outputs)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    subject_labels = [data["subject_labels"] for data in batch]
    object_labels = [data["object_labels"] for data in batch]
    subject_ids = [data["subject_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    subject_labels_padded = padding(subject_labels, max_length)
    object_labels_padded = padding(object_labels, max_length)
    subject_ids = np.array(subject_ids)

    return torch.tensor(token_ids_padded, dtype=torch.long), torch.tensor(token_type_ids_padded, dtype=torch.float32), \
        torch.tensor(subject_labels_padded, dtype=torch.long), torch.tensor(object_labels_padded, dtype=torch.long),\
        torch.tensor(subject_ids, dtype=torch.long)

class ExtractTrainer:
    def __init__(self):
        # 加载数据
        data_path = "./corpus/三元组抽取/train_data.json"
        self.vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置
        self.data = load_data(data_path)
        self.model_name = "roberta" # 选择模型名字
        self.model_path = "./state_dict/roberta_wwm_pytorch_model.bin" # roberta模型位置
        self.recent_model_path = "" # 用于把已经训练好的模型继续训练
        self.model_save_path = "./bert_model_relation_extrac.bin"
        self.batch_size = 16
        self.lr = 1e-5
        # 加载字典
        self.word2idx = load_chinese_base_vocab(self.vocab_path)
        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(self.vocab_path, model_name=self.model_name, model_class="relation_extrac", target_size=len(predicate2id))
        ## 加载预训练的模型参数～
        load_model_params(self.bert_model, self.model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.to(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=self.lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = ExtractDataset(self.data, self.vocab_path)
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
        report_loss = 0.0
        for token_ids, token_type_ids, subject_lables, object_labels, subject_ids in tqdm(dataloader,position=0, leave=True):
            step += 1
            if step % 300 == 0:
                print("report loss is " + str(report_loss))
                report_loss = 0.0
            
            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                subject_ids,
                                                subject_labels=subject_lables,
                                                object_labels=object_labels,
                                                device=self.device
                                                )
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bert_model.parameters(), 5.0)
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()
            report_loss += loss.item()
        
        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch)+". loss is " + str(total_loss) + ". spend time is "+ str(spend_time))
        # 保存模型
        self.save(self.model_save_path)

if __name__ == "__main__":

    trainer = ExtractTrainer()
    train_epoches = 10
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)

    # data = load_data("./corpus/三元组抽取/train_data.json")
    # # print(data[:5])
    # vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置
    # datasets = ExtractDataset(data, vocab_path)
    # word2idx = load_chinese_base_vocab(vocab_path)
    # # self.idx2word = {k: v for v, k in self.word2idx.items()}
    # tokenizer = Tokenizer(word2idx)
    # dataloader = DataLoader(datasets, shuffle=True, collate_fn=collate_fn, batch_size=2)
    # for token_ids, segment_ids, subject_labels, object_labels, subject_ids in dataloader:
    #     print(token_ids)
    #     print(segment_ids)
    #     print(subject_ids)
    #     print(subject_labels)
    #     print(object_labels)
    #     print(tokenizer.decode(token_ids[0]))
    #     print(tokenizer.decode(token_ids[1]))
    #     break