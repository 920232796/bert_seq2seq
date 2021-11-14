## roberta-large 自动摘要的例子
import torch
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert
import glob 

vocab_path = "./state_dict/roberta_wwm_vocab.txt"  # roberta模型字典的位置
word2idx = load_chinese_base_vocab(vocab_path, simplfied=False)
model_name = "roberta-large"  # 选择模型名字
model_path = "./state_dict/roberta-large/pytorch_model.bin"  # 模型位置
model_save_path = "./state_dict/bert_auto_gen_model.bin"
batch_size = 4
lr = 1e-5


data_path = glob.glob("./corpus/THUCNews/*/*.txt")

class BertDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self) :
        ## 一般init函数是加载所有数据
        super(BertDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir
        
        self.idx2word = {k: v for v, k in word2idx.items()}
        self.tokenizer = Tokenizer(word2idx)

    def __getitem__(self, i):
        ## 得到单个数据
        file_path = data_path[i]
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        title = lines[0].strip("\n")
        content = lines[1:]
        content = "".join(content)
        content = content.replace(" ", "").replace("\t", "").replace("\n", "").replace("”", "").replace("“", "").replace("　　", "")
        content = content.split("。")
        cons_text = ""
        index = 0
        while len(cons_text) < 400 and index < len(content):
            cons_text += content[index] + "。"
            index += 1
        # print(title)
        # print(cons_text)
        # print(content)
        if len(title) + len(content) > 500:
            return self.__getitem__(i + 1)
        
        if len(cons_text) == 0:
            return self.__getitem__(i + 1)

        token_ids, token_type_ids = self.tokenizer.encode(title, cons_text, max_length=512)
        
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
        }
        return output

    def __len__(self):
        return len(data_path)
        
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

class Trainer:
    def __init__(self):
        # 加载数据
       
      
        # self.sents_src= torch.load("./corpus/auto_title/train_clean.src")
        # self.sents_tgt = torch.load("./corpus/auto_title/train_clean.tgt")

        # 判断是否有可用GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(word2idx, model_name=model_name, model_class="seq2seq")
        self.bert_model.set_device(self.device)
        ## 加载预训练的模型参数～  
        self.bert_model.load_pretrain_params(model_path)

        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = BertDataset()
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
        for token_ids, token_type_ids, target_ids in tqdm(dataloader,position=0, leave=True):
            step += 1
            if step % 300 == 0:
                self.bert_model.eval()
                test_data = ["国足大胜意大利。",
                 "阿里巴巴股票大涨。", 
                 "特斯拉发布第三季度财报。"]
                with open("./res_gen_article.txt", "a+", encoding="utf-8") as f :
                    for text in test_data:
                        out = self.bert_model.sample_generate(text, out_max_length=400, top_k=30, max_length=512)
                        f.write(out)
                        f.write("\n")
                
                    print("loss is " + str(report_loss))
                    f.write(f"loss is {report_loss} \n ")


                
                report_loss = 0
                # self.eval(epoch)
                self.bert_model.train()
            if step % 8000 == 0:
                self.save(model_save_path)

            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                token_type_ids,
                                                labels=target_ids,
                                                )
            report_loss += loss.item()
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
    train_epoches = 20

    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)
