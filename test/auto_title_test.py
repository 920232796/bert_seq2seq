import torch
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert

auto_title_model = "./state_dict/bert_auto_title_model2.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    vocab_path = "./state_dict/roberta_wwm_vocab.txt"  # roberta模型字典的位置
    model_name = "roberta"  # 选择模型名字
    # 加载字典
    word2idx, keep_tokens = load_chinese_base_vocab(vocab_path, simplfied=True)
    # 定义模型
    bert_model = load_bert(word2idx, model_name=model_name)
    bert_model.set_device(device)
    bert_model.eval()
    ## 加载训练的模型参数～
    bert_model.load_all_params(model_path=auto_title_model, device=device)

    test_data = ["针对央视3·15晚会曝光的电信行业乱象，工信部在公告中表示将严查央视3·15晚会曝光通信违规违法行为，工信部称已约谈三大运营商有关负责人并连夜责成三大运营商和所在省通信管理局进行调查依法依规严肃处理",
                "楚天都市报记者采访了解到，对于进口冷链食品，武汉已经采取史上最严措施，进行“红区”管理，严格执行证明查验制度，确保冷冻冷藏肉等冻品的安全。",
                "新华社受权于18日全文播发修改后的《中华人民共和国立法法》修改后的立法法分为“总则”“法律”“行政法规”“地方性法规自治条例和单行条例规章”“适用与备案审查”“附则”等6章共计105条"]
    for text in test_data:
        with torch.no_grad():
            print(bert_model.generate(text, beam_size=3))



