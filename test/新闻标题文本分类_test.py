import torch 
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert

target = ["财经", "彩票", "房产", "股票", "家居", "教育", "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"]

cls_model = "./state_dict/bert_multi_classify_model.bin"
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
    test_data = ["编剧梁馨月讨稿酬六六何念助阵 公司称协商解决", 
                "西班牙BBVA第三季度净利降至15.7亿美元", 
                "基金巨亏30亿 欲打开云天系跌停自救"]
    for text in test_data:
        with torch.no_grad():
            text, text_ids = tokenizer.encode(text)
            text = torch.tensor(text, device=device).view(1, -1)
            print(target[torch.argmax(bert_model(text)).item()])
        


