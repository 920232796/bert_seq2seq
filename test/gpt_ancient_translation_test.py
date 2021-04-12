import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")

import torch
from bert_seq2seq.utils import load_gpt
from bert_seq2seq.tokenizer import load_chinese_base_vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_path = "./state_dict/gpt2通用中文模型/vocab.txt"
model_path = "./state_dict/gpt_ancient_trans_model.bin"

if __name__ == "__main__":
    word2ix = load_chinese_base_vocab(vocab_path)
    model = load_gpt(word2ix)
    model.eval()
    model.set_device(device)
    model.load_all_params(model_path)

    print(model.sample_generate("自昔羲后，因以物命官，事简人淳，唯以道化，上无求欲於下，下无干进於上，百姓自足，海内乂安，不是贤而非愚，不沽名而尚行，推择之典，无所闻焉。", out_max_length=300, add_eos=True))