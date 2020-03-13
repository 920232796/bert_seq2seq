# bert_seq2seq
pytorch实现bert做seq2seq任务，使用unilm方案。注意本项目可以做bert seq2seq 任何任务，比如对联，写诗，自动摘要等等等等，只要你下载数据集，并且写好对应trainer.py，即可。

部分代码参考了 https://github.com/huggingface/transformers/ 和 https://github.com/bojone/bert4keras 
非常感谢！！！


### 运行
1. 下载想训练的数据集，可以专门建个corpus文件夹存放，参考config.py里面的一些配置。
2. 下载bert中文预训练权重 "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
3. 下载bert中文字典 "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
4. 去train文件夹下面运行对应的trainer.py，针对不同任务，运行不同trainer.py文件。

### 效果
想看效果，可以去我网站～ http://www.blog.zhxing.online/#/readBlog?blogId=315 
多谢支持。
