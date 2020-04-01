# bert_seq2seq
一个轻量级的小框架。

pytorch实现bert做seq2seq任务，使用unilm方案。注意本项目可以做bert seq2seq 任何任务，比如对联，写诗，自动摘要等等等等，只要你下载数据集，并且写好对应train.py，即可，只需要改动很少代码，便可以重新训练新任务，如果喜欢的话欢迎star～ 如果遇到问题也可以提issue，保证会回复。

部分代码参考了 https://github.com/huggingface/transformers/ 和 https://github.com/bojone/bert4keras 
非常感谢！！！
### 目前支持
目前支持model_name 为 roberta 或者 bert
### 安装 
1. ```pip install bert-seq2seq```
2. 安装pytorch 
3. 安装tqdm 可以用来显示进度条 ```pip install tqdm```
### 运行
1. 下载想训练的数据集，可以专门建个corpus文件夹存放。
2. 使用roberta模型，模型和字典文件需要去 https://drive.google.com/file/d/1iNeYFhCBJWeUsIlnW_2K6SMwXkM4gLb_/view 这里下载。 具体可以参考这个github仓库～ https://github.com/ymcui/Chinese-BERT-wwm
3. 如果使用普通的bert模型，下载bert中文预训练权重 "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin", 下载bert中文字典 "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt".
4. 去example文件夹下面运行对应的trainer.py，针对不同任务，运行不同train.py文件，进行训练。
5. 每次对于一个新的任务，只需要改动很少一部分代码，配置好模型位置，字典位置，写好数据处理构造输入输出(也就是read_corpus函数)即可。举个例子：
```python
class PoemTrainer:
    def __init__(self):
        # 加载数据
        data_dir = "./corpus/Poetry"
        self.vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置
        self.sents_src, self.sents_tgt = read_corpus(data_dir, self.vocab_path)
        self.model_name = "roberta" # 选择模型名字
        self.model_path = "./state_dict/roberta_wwm_pytorch_model.bin" # roberta模型位置
        self.recent_model_path = "" # 用于把已经训练好的模型继续训练
        self.model_save_path = "./bert_model.bin" #训练好的模型保存在哪
        self.batch_size = 16
        self.lr = 1e-5
```
在对应的train文件里面，只要配置好这些必要的信息，基本就可以开始训练了。开始一个新任务只需要10分钟改代码的时间。

### 效果
效果感觉还是很不错的～ 
#### 写诗
![image.png](http://www.zhxing.online/image/acb592f918894ca6b62435d2464d3cb0.png)
#### 对联
![image.png](http://www.zhxing.online/image/42eec322d6cc419da0efdc45c02d9f25.png)
![image.png](http://www.zhxing.online/image/25c1967ecfb14c5c9e68da7e3615ccf5.png)

![image.png](http://www.zhxing.online/image/540a4f1be41d4a3cbd2ccf1b26895868.png)


想看文章，可以去我网站～ http://www.blog.zhxing.online/#/readBlog?blogId=315 
多谢支持。另外，网站上面还有一些介绍unilm论文和特殊的mask如何实现的文章，可以去网站里搜索一下。http://www.blog.zhxing.online/#/

### 更新记录
2020.04.01: 添加了写诗的task

2020.04.01: 重构了代码，开始训练一个新的任务花费时间更少。

python setup.py sdist
twine upload dist/bert_seq2seq-0.0.1.tar.gz

