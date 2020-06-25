
import torch 
import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")
from bert_seq2seq.model.bert_model import BertModel, BertConfig
from bert_seq2seq.tokenizer import load_chinese_base_vocab

if __name__ == "__main__":
    # word2ix = load_chinese_base_vocab("./state_dict/bert-base-chinese-vocab.txt")
    # config = BertConfig(len(word2ix))
    # model = BertModel(config)

    check_point = torch.load("./state_dict/bert-base-chinese-pytorch_model.bin")
    
    for k, v in check_point.items():
        print(k)