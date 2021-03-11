import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")

import torch 
from bert_seq2seq.utils import load_gpt
from bert_seq2seq.tokenizer import load_chinese_base_vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    word2ix = load_chinese_base_vocab("./state_dict/gpt_vocab.txt")
    model = load_gpt(word2ix)
    model.eval()
    model.set_device(device)
    model.load_pretrain_params("./state_dict/gpt_pytorch_model.bin")

    print(model.sample_generate("今天天气好", out_max_length=300))