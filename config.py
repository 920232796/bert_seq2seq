##通用配置
bert_chinese_model_path = "./state_dict/bert-base-chinese-pytorch_model.bin"
base_chinese_bert_vocab = "./state_dict/bert-base-chinese-vocab.txt"
max_length=256


## 情感分析相关配置
sentiment_train_corpus_dir = "./corpus/sentiment_data"
# test_corpus_path = ./corpus/hotel_feedbacks_sentiment/test_data.txt
sentiment_batch_size = 16
sentiment_lr = 0.0001



## 周公解梦数据路径
dream_train_corpus_path = "./corpus/dream/周公解梦数据.csv"

## 自动标题数据路径
auto_title_train_path = "./corpus/auto_title_data/data.txt"

## 对联数据文件夹路径
duilian_corpus_dir = "./corpus/对联"