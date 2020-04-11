import torch 
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F 

# target = torch.tensor([[0], [1], [2], [3]])
# print(F.one_hot(target, 4))

# t1 = torch.tensor([[0, 1, 0]]) # 表示形容词
# t2 = torch.tensor([[0, 0, 1]]) # 表示名词

# w = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# res = torch.einsum("bi,ij,bj->b",t1, w, t2)
# print(res)

# t1 = torch.tensor([[[0, 1, 0], [0, 1, 0]]])
# t2 = torch.tensor([[[0, 1, 0],[1, 0, 0]]])
# print(torch.einsum("bni,bni->b", t1, t2))

# t1 = torch.tensor([[1], [2], [3]])
# t2 = torch.tensor([[1, 2, 3], [4, 5, 6], [2, 2, 2]])
# print(torch.sum(t1 + t2, dim=1))

def logsumexp(x, dim=None, keepdim=False):
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    out = xm + torch.log(torch.sum(torch.exp(x - xm), dim=dim, keepdim=True))
    return out if keepdim else out.squeeze(dim)

def err_logsumexp(x, dim=None, keepdim=False):
    if dim is None:
        x, dim = x.view(-1), 0
    out = torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))
    return out if keepdim else out.squeeze(dim)

# t1 = torch.tensor([[0, 0, 0], [-1000, -1000, -1000]], dtype=torch.float64)
# print(err_logsumexp(t1, 1))


# print(logsumexp(t1, 1))

# t2 = torch.exp(torch.tensor([-1000000000000], dtype=torch.float32))
# print(t2)

# print(torch.tensor([0.00000000000000000000000000000000000000000000001], dtype=torch.float32))

# a1 = np.array([[0, 0, 0], [0, 0, float("inf")]])
# print(np.log(np.sum(np.exp(a1), axis=1)))


# from bert_seq2seq.utils import load_bert, load_recent_model
# from bert_seq2seq.tokenizer import load_chinese_base_vocab, Tokenizer

# target = ["pad", "O", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG"]

# vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置
# word2idx = load_chinese_base_vocab(vocab_path)
# tokenier = Tokenizer(word2idx)

# bert_model = load_bert(vocab_path, model_class="sequence_labeling", target_size=len(target))

# bert_model.load_state_dict(torch.load("./state_dict/bert_ner_model.bin", map_location="cpu"))

# bert_model.eval()
# test_data = ["北京烤鸭好不好吃，只有真的去北京吃过才知道。"]
# for text in test_data:
#     text_ids, _ = tokenier.encode(text)
#     text_ids = torch.tensor(text_ids).view(1, -1)
#     out = bert_model(text_ids).squeeze(0)
#     out_target = torch.argmax(out, dim=-1)
#     decode_target = [target[i.item()] for i in out_target]
#     print(decode_target)
#     res = {"LOC": [], "ORG": [], "PER": []}
#     for index, i in enumerate(decode_target):
#         # print(i)
#         if i != "O":
#             # print(i[-3: ])
#             res[i[-3:]].append(text[index - 1])
    
#     print(res)


