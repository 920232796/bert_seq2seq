import unicodedata
from pathlib import Path 
import torch 
from model.bert_model import BertConfig, BertModel
from tokenizer import load_chinese_base_vocab
from typing import List, Dict
import torch.nn as nn 


if __name__ == "__main__":

    # ## 测试一下加载模型～
    # word2idx = load_chinese_base_vocab()
    # print(word2idx["[PAD]"])
    # bert_config = BertConfig(len(word2idx))
    # bert = BertModel(bert_config)

    # for k, v in bert.named_parameters():
    #     if k == "encoder.layer.11.attention.output.LayerNorm.gamma":
    #         print(v)

    # bert_chinese_path = "./state_dict/bert-base-chinese-pytorch_model.bin"

    # checkpoint = torch.load(bert_chinese_path, map_location="cpu")
    # checkpoint = {k[5:]: v for k, v in checkpoint.items() if k[:4] == "bert" and "pooler" not in k}
    # bert.load_state_dict(checkpoint, strict=False)

    # for k, v in bert.named_parameters():
    #     if k == "encoder.layer.11.attention.output.LayerNorm.gamma":
    #         print(v)

    #~~~~~~~~~~~~~~~~~

    # t1 = torch.tensor([[1, 2, 3, 4]])
    # t2 = torch.tensor([[1], [2], [3], [4]])
    # print(t1 * t2)

    ## 测试loss mask计算

    # 传入 segment id （batch-size， seq—len）
    # s = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1]])
    # print(s.shape)
    # seq_len = s.shape[1]
    # ones = torch.ones((1, 1, seq_len, seq_len))
    # a_mask = ones.tril()
    # s_ex12 = s.unsqueeze(1).unsqueeze(2)
    # s_ex13 = s.unsqueeze(1).unsqueeze(3)
    # print(s_ex12.shape)
    # print(s_ex13.shape)
    # a_mask = (1 - s_ex12) * (1 - s_ex13) + s_ex13 * a_mask 

    # print(a_mask)

    # 测试成功，写一篇文章。

    # 还差 最后的target mask 如何计算？？ 计算loss只求目标那几个值的loss 忽略掉别的部分的loss
    # 可以直接取 segment_id [:, 1:] 这样就行了啊！！我的天。。

    # import numpy as np 

    # a1 = np.empty((1, 0), dtype=int)
    # print(a1.shape)


    # t1 = torch.rand(1, 3)
    # print(t1)

    # print(t1.repeat(3, 1))


    # t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # t1 = torch.cat((t1, torch.tensor([[1, 1, 1]])), dim=0)
    # print(t1)

    # t1 = torch.empty((3, 0), dtype=torch.long)
    # t2 = torch.tensor([[1], [2], [3]])
    # t3 = torch.cat((t1, t2), dim=1)
    # print(t3)

    # indice1 = torch.tensor([1, 0, 2])
    # t1 = torch.empty((3, 0), dtype=torch.long)
    # t2 = torch.tensor([[1], [2], [3]])
    # t3 = torch.cat((t1[indice1], t2), dim=1)
    # print(t3)

    # t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # t2, t3 = torch.topk(t1.view(-1), 3)
    # print(t3)
    # print(torch.argmax(t3))

    # t1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    # t1 = nn.Softmax(dim=-1)(t1)
    # print(t1)
    token_type_id = torch.tensor([[0, 0, 1, 1, 0, 0]])
    seq_len = 6
    ones = torch.ones((1, 1, seq_len, seq_len))
    a_mask = ones.tril() # 下三角矩阵
    s_ex12 = token_type_id.unsqueeze(1).unsqueeze(2)
    s_ex13 = token_type_id.unsqueeze(1).unsqueeze(3)
    a_mask = (1 - s_ex12) * (1 - s_ex13) + s_ex13 * a_mask 

    # print(a_mask)

    print(a_mask * torch.tensor([[1, 1, 1, 1, 0, 0]]))

   
    t1 = torch.tensor([1, 2, 3, 0])
    # t2 = t1.unsqueeze(0)
    # print(t2)
    # t3 = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
    # print(t2 + t3)
    mask = t1.eq(0)
    mask = 1 - mask.long()
    print(mask)
    mask1 = (t1 > 0).long()
    print(mask1)
    # t1.masked_fill_(mask, 100)
    # print(t1)

    loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    input1 = torch.tensor([[1, 2, 3], [1, 1, 1]], dtype=torch.float32).view(-1, 3)
    label = torch.tensor([[1], [0]]).view(-1)

    print(loss(input1, label).sum())

