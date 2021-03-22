from re import I
import torch 
import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")
from bert_seq2seq.utils import load_bert
import torch.nn.functional as F 

from bert_seq2seq.bert_cls_classifier_sigmoid import BertClsClassifier
from bert_seq2seq.model.bert_model import BertConfig
from bert_seq2seq.tokenizer import load_chinese_base_vocab

if __name__ == '__main__':
    # config = BertConfig(vocab_size=1000)
    word2ix = load_chinese_base_vocab("./state_dict/roberta_wwm_vocab.txt")
    model = BertClsClassifier(word2ix, target_size=1, model_name="")

