import torch 
import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert

target = ["0", "1"]

cls_model = "./state_dict/bert_semantic_matching.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    vocab_path = "./state_dict/roberta_wwm_vocab.txt"  # roberta模型字典的位置
    model_name = "roberta"  # 选择模型名字
    # 加载字典
    word2idx = load_chinese_base_vocab(vocab_path, simplfied=False)
    tokenizer = Tokenizer(word2idx)
    # 定义模型
    bert_model = load_bert(word2idx, model_name=model_name, model_class="cls", target_size=len(target))
    bert_model.set_device(device)
    bert_model.eval()
    ## 加载训练的模型参数～
    bert_model.load_all_params(model_path=cls_model, device=device)
    test_data = ["你是不是我仇人#你是俺的仇人吗",
                "这个就没意思了#我没别的意思", 
                "查一下我的家在哪里#家在哪里?"]
    for text in test_data:
        with torch.no_grad():
            text_ids, _ = tokenizer.encode(text)
            text_ids = torch.tensor(text_ids, device=device).view(1, -1)
            print(text + " -> res is " + str(target[torch.argmax(bert_model(text_ids)).item()]))
