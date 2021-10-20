import torch
from bert_seq2seq.tokenizer import  load_chinese_base_vocab, T5PegasusTokenizer
from bert_seq2seq.extend_model_method import ExtendModel
import glob 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from bert_seq2seq.t5_ch import T5Model

vocab_path = "./state_dict/t5-chinese/vocab.txt"
model_path = "./state_dict/t5_autotile.bin"
word2idx = load_chinese_base_vocab(vocab_path)

model = T5Model(word2idx, size="base")
model.set_device(device)
model.load_all_params(model_path)
model.eval()

all_txt = glob.glob("./*.txt")
print(all_txt)
for t in all_txt:
    with open(t, encoding="utf-8") as f:
        content = f.read()
        out = model.sample_generate_encoder_decoder(content)
        print(out)

# for t in all_txt:




