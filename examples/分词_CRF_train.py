## http://sighan.cs.uchicago.edu/bakeoff2005/ 数据使用pku的分词数据
import torch
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert
import re

data_path = "./corpus/分词/pku_training.utf8"
vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置
model_name = "roberta" # 选择模型名字
model_path = "./state_dict/roberta_wwm_pytorch_model.bin" # roberta模型位置
model_save_path = "./bert_word_segmentation_model.bin"
batch_size = 8
lr = 1e-5
crf_lr = 1e-2
maxlen = 512

word2idx = load_chinese_base_vocab(vocab_path)
tokenier = Tokenizer(word2idx)
target = [0, 1, 2, 3]

def read_corpus():
    """
    读原始数据
    """
    D = []
    with open(data_path, encoding='utf-8') as f:
        for l in f:
            D.append(re.split(' +', l.strip()))
    return D

## 自定义dataset
class WSDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, sents_src) :
        ## 一般init函数是加载所有数据
        super().__init__()
        # 读原始数据
        self.sents_src = sents_src

        self.idx2word = {k: v for v, k in word2idx.items()}

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.sents_src[i]
        token_ids = [tokenier.token_start_id]
        label = [0]
        for word in src:
            w_token_ids = tokenier.encode(word, is_segment=False)[1:-1]
            if len(token_ids) + len(w_token_ids) > maxlen:
                break

            token_ids += w_token_ids
            if len(w_token_ids) == 1:
                label += [0]
            else:
                label += [1] + [2] * (len(w_token_ids) - 2) + [3]

        token_ids += [tokenier.token_end_id]
        label += [0]

        while len(token_ids) > 512:
            token_ids.pop(-2)
            label.pop(-2)

        token_type_ids = [0] * len(token_ids)


        # print(token_ids)
        # print(label)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
            "target_id": label
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

    trans = model.state_dict()["crf_layer.trans"]
    for text in test_data:
        text_encode, text_ids = tokenier.encode(text)
        tokens = tokenier.tokenize(text)
        mapping = tokenier.rematch(text, tokens)
        text_tensor = torch.tensor(text_encode, device=device).view(1, -1)
        out = model(text_tensor).squeeze(0) # 其实是nodes
        labels = viterbi_decode(out, trans)
        decode_res = []
        start = 0
        for i in range(len(labels)):
            map_i = mapping[i]
            if len(map_i) < 1:
                continue
            if labels[i] == 0:
                # 说明是单字
                start = 0
                decode_res += ["".join([text[j] for j in map_i])]
            elif start == 1:
                ## 证明还有字没加到最后
                decode_res[-1] = decode_res[-1] + "".join([text[j] for j in map_i])
                if labels[i] == 3:
                    # 结束了
                    start = 0
            else :
                start = 1
                decode_res += ["".join([text[j] for j in map_i])]
        print(decode_res)

class Trainer:
    def __init__(self):
        # 加载数据
        
        self.sents_src = read_corpus()
        # 判断是否有可用GPU
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(word2idx, model_name=model_name, model_class="sequence_labeling_crf", target_size=len(target))
        ## 加载预训练的模型参数～
        self.bert_model.load_pretrain_params(model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.set_device(self.device)
        # 声明需要优化的参数
        # 声明需要优化的参数
        crf_params = list(map(id, self.bert_model.crf_layer.parameters()))  ## 单独把crf层参数拿出来
        base_params = filter(lambda p: id(p) not in crf_params, self.bert_model.parameters())
        self.optimizer = torch.optim.Adam([
            {"params": base_params},
            {"params": self.bert_model.crf_layer.parameters(), "lr": crf_lr}], lr=lr, weight_decay=1e-3)

        # 声明自定义的数据加载器
        dataset = WSDataset(self.sents_src)
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
        start_time = time.time()
        step = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader,position=0, leave=True):
            # print(target_ids.shape)
            step += 1
            if step % 200 == 0:
                test_data = ["日寇在京掠夺文物详情。", "以书结缘，把欧美，港台流行的食品类食谱汇集一堂。", "明天天津下雨，不知道主任还能不能来学校吃个饭。", "中国政府将继续坚持奉行独立自主的和平外交政策"]
                ner_print(self.bert_model, test_data)
                self.bert_model.train()

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
        self.save(model_save_path)

if __name__ == '__main__':
    
    trainer = Trainer()
    train_epoches = 25
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)

