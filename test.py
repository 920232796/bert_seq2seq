import torch 

## 测试加载一下 roberta 模型
from model.roberta_model import BertConfig, BertModel
from tokenizer import load_chinese_base_vocab

if __name__ == "__main__":
    # checkpoint = torch.load("./state_dict/roberta_wwm_pytorch_model.bin")
    # word2idx = load_chinese_base_vocab()
    # print(len(word2idx))
    # bert_config = BertConfig(len(word2idx))
    # bert = BertModel(bert_config)

    # # checkpoint = {k[5:]: v for k, v in checkpoint.items()
    # #                                         if k[:4] == "bert" and "pooler" not in k}
    # same_list = []
    # for k, v in checkpoint.items():
    #     # print(k)
    #     same_list.append(k[5:])

    # for kk, vv in bert.named_parameters():
    #     # print(kk)
    #     if kk in same_list:
    #         print(kk)

    # t1 = torch.tensor([[1, 2, 0], [-1, 4, 6]])
    # t2 = (t1 > 0).float()

    # t3 = t2.unsqueeze(1).unsqueeze(2)
    # print(t3.shape)

    # t1 = torch.tensor([[1, 1,   1, 0]])
    # t3 = torch.tensor([[1], [1], [1], [0]])
    # print(t1 * t3)
    # print(t1.shape)
    # t2 = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

    # print(t1 * t3 * t2 )

    # str1 = "躺汤糖堂"
    # l1 = [word for word in str1]
    # print(l1)
    # if "糖" in l1:
    #     print("hah")

    # l1 = [1, 2, 3, 4, 6]
    # print(l1[:4] + l1[4 + 1:])

    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(t1.shape)
    t2 = torch.tensor([1, 2])
    print(t2.shape)
    # print(t1.view(-1, 1) + t2)
    print(t1 + t2.view(-1, 1))