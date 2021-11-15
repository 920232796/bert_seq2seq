## gpt2 进行文言文翻译
from torch.utils import data
from bert_seq2seq import load_gpt, tokenizer
import torch
from tqdm import tqdm
import os 
import time
import  glob
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import Tokenizer, load_chinese_base_vocab


vocab_path = "./state_dict/gpt2/vocab.txt"
model_path = "./state_dict/gpt2/pytorch_model.bin"

model_save_path = "./gpt2_article_gen.bin"
batch_size = 4
lr = 2e-5
word2idx = load_chinese_base_vocab(vocab_path)



data_path = glob.glob("./corpus/THUCNews/*/*.txt")


class SeqDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self):
        ## 一般init函数是加载所有数据
        super(SeqDataset, self).__init__()
        # 读原始数据
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
        content = content.replace("\n", "&").replace("　　", "").replace("&&", "").replace("”", "").replace("“", "")
        content = content.split("&")
        cons_text = ""
        index = 0
        while len(cons_text) < 900 and index < len(content):
            cons_text += content[index] + "&"
            index += 1
        # print(title)
        # print(cons_text)
        # # print(content)328
        if len(title) + len(content) > 1024:
            if i == 0:
                return self.__getitem__(i + 1)
            else :
                return self.__getitem__(i - 1)
        
        if len(cons_text) == 0:
            if i == 0:
                return self.__getitem__(i + 1)
            else :
                return self.__getitem__(i - 1)

        token_ids, _ = self.tokenizer.encode(title + "&" + cons_text, max_length=1000)
        
        output = {
            "token_ids": token_ids,
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

    token_ids_padded = padding(token_ids, max_length)
    target_ids_padded = token_ids_padded.clone()
    target_ids_padded[target_ids_padded == 0] = -100

    return token_ids_padded, target_ids_padded


class Trainer:
    def __init__(self):
        # 加载数据
        # 判断是否有可用GPU
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        # self.gpt_model = AutoModelWithLMHead.from_pretrained(model_path)
        # self.gpt_model.to(self.device)
        # self.gpt_model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(model_path)
        self.gpt_model = load_gpt(word2idx)
        ## 加载预训练的模型参数～
        self.gpt_model.load_pretrain_params(model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.gpt_model.set_device(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.gpt_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-5)
        # 声明自定义的数据加载器
        dataset = SeqDataset()
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def train(self, epoch):
        # 一个epoch的训练
        self.gpt_model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)

    def save(self, save_path):
        """
        保存模型
        """
        self.gpt_model.save_all_params(save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        report_loss = 0
        start_time = time.time()  ## 得到当前时间
        step = 0
        for token_ids, target_ids in tqdm(dataloader, total=len(dataloader)):
        # for token_ids, target_ids in tqdm(dataloader:
            step += 1
        
            if step % 1000 == 0:
                print(f"epoch is {epoch}")
                self.gpt_model.eval()
                # self.gpt_model.to(torch.device("cpu"))
                # text_generator = TextGenerationPipeline(self.gpt_model, tokenizer)   
                test_data = ["尚品宅配：家具定制模范生。", "今天的天气还不错。", "受双十一影响，阿里巴巴股票今天大涨。"]
                for text in test_data:
                    # out = text_generator(text, max_length=300, do_sample=True)
                    # print(out)
                    print(self.gpt_model.sample_generate(text, add_eos=False, top_k=10, out_max_length=900, top_p=0.7, temperature=3.0, repetition_penalty=1.5))
                self.gpt_model.train()
                # self.gpt_model.to(self.device)
                print("report loss is " + str(report_loss))
                report_loss = 0.0
                self.gpt_model.save_all_params(model_save_path)
                print("模型保存完毕")

            # 因为传入了target标签，因此会计算loss并且返回
            loss, _ = self.gpt_model(token_ids,
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

        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch) + ". loss is " + str(total_loss) + ". spend time is " + str(spend_time))
        # 保存模型
        self.save(model_save_path)


if __name__ == '__main__':

    trainer = Trainer()
    train_epoches = 5
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)
