import torch
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert

auto_title_model = "./state_dict/bert_model_poem_ci_duilian.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    vocab_path = "./state_dict/roberta_wwm_vocab.txt"  # roberta模型字典的位置
    model_name = "roberta"  # 选择模型名字
    # model_path = "./state_dict/bert-base-chinese-pytorch_model.bin"  # roberta模型位
    # 加载字典
    word2idx = load_chinese_base_vocab(vocab_path, simplfied=False)
    # 定义模型
    bert_model = load_bert(word2idx, model_name=model_name)
    bert_model.set_device(device)
    bert_model.eval()
#   ## 加载预训练的模型参数～
    checkpoint = torch.load(auto_title_model, map_location="cpu")
    # print(checkpoint)
    bert_model.load_all_params(model_path=auto_title_model, device=device)
    test_data = ["江山竞秀，万里风光入画图##对联"]
    with torch.no_grad():
        for text in test_data:
            if text[-1] == "句" or text[-1] == "诗":
                print(bert_model.generate(text, beam_size=3, is_poem=True))
            else:
                print(bert_model.generate(text, beam_size=3, is_poem=False))



