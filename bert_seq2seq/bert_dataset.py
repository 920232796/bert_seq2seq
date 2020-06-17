# import torch 
# from torch.utils.data import Dataset, DataLoader
# from bert_seq2seq.tokenizer import load_chinese_base_vocab, Tokenizer

# ## 自定义dataset
# class BertDataset(Dataset):
#     """
#     针对特定数据集，定义一个相关的取数据的方式
#     """
#     def __init__(self, sents_src, sents_tgt, vocab_path) :
#         ## 一般init函数是加载所有数据
#         super(BertDataset, self).__init__()
#         # 读原始数据
#         # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
#         self.sents_src = sents_src
#         self.sents_tgt = sents_tgt
#         self.word2idx = load_chinese_base_vocab(vocab_path)
#         self.idx2word = {k: v for v, k in self.word2idx.items()}
#         self.tokenizer = Tokenizer(self.word2idx)

#     def __getitem__(self, i):
#         ## 得到单个数据
#         # print(i)
#         src = self.sents_src[i]
#         tgt = self.sents_tgt[i]
#         token_ids, token_type_ids = self.tokenizer.encode(src, tgt)
#         output = {
#             "token_ids": token_ids,
#             "token_type_ids": token_type_ids,
#         }
#         return output

#     def __len__(self):

#         return len(self.sents_src)

# def collate_fn(batch):
#     """
#     动态padding， batch为一部分sample
#     """
#     def padding(indice, max_length, pad_idx=0):
#         """
#         pad 函数
#         注意 token type id 右侧pad是添加1而不是0，1表示属于句子B
#         """
#         pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
#         return torch.tensor(pad_indice)
   
#     token_ids = [data["token_ids"] for data in batch]
#     max_length = max([len(t) for t in token_ids])
#     token_type_ids = [data["token_type_ids"] for data in batch]

#     token_ids_padded = padding(token_ids, max_length)
#     token_type_ids_padded = padding(token_type_ids, max_length)
#     target_ids_padded = token_ids_padded[:, 1:].contiguous()

#     return token_ids_padded, token_type_ids_padded, target_ids_padded

