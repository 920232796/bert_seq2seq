
import torch 
from bert_seq2seq import load_gpt
from bert_seq2seq import load_chinese_base_vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_path = "./state_dict/gpt2通用中文模型/vocab.txt"
model_path = "./state_dict/gpt2通用中文模型/pytorch_model.bin"

if __name__ == "__main__":
    word2ix = load_chinese_base_vocab(vocab_path)
    model = load_gpt(word2ix)
    model.eval()
    model.set_device(device)
    model.load_pretrain_params(model_path)

    print(model.sample_generate("今天天气好", out_max_length=300))