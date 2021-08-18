## 英文自动摘要测试文件
import torch
import glob
import json
from rouge import Rouge
from bert_seq2seq import load_bert
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
word2idx = tokenizer.get_vocab()
auto_title_model = "./state_dict/bert_english_auto_title_model.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxlen = 256

if __name__ == "__main__":
    model_name = "bert"  # 选择模型名字
    # 定义模型
    bert_model = load_bert(word2idx, tokenizer=tokenizer, model_name=model_name)
    bert_model.set_device(device)
    bert_model.eval()
    ## 加载训练的模型参数～
    bert_model.load_all_params(model_path=auto_title_model, device=device)
    rouge = Rouge()
    test_file = glob.glob("./corpus/english_autotitle_test/*.json")
    num_file = len(test_file)
    rouge_1_item = [0.0, 0.0, 0.0]
    with open("./auto_title_res.txt", "a+") as fw:
        for s_file in test_file :
            with open(s_file, "r") as f:
                c = f.read()
                j = json.loads(c)
                title = j["Title"]
                text = j["abstract"]
                out = bert_model.generate(text, beam_size=3, out_max_length=100, max_length=maxlen)
                print(out)
                fw.write(title + "\t" + out + "\t" + text + "\n")

                rouge_score = rouge.get_scores(title, out)
                print(rouge_score)
                rouge_1 = rouge_score[0]["rouge-1"]
                rouge_1_item[0] += rouge_1["f"]
                rouge_1_item[1] += rouge_1["p"]
                rouge_1_item[2] += rouge_1["r"]
                # print(rouge_score[0]["rouge-2"])
                # print(rouge_score[0]["rouge-l"])
    for i in range(len(rouge_1_item)):
        rouge_1_item[i] = rouge_1_item[i] / num_file 

            
    print(rouge_1_item)




