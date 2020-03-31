# bert_seq2seq
pytorch实现bert做seq2seq任务，使用unilm方案。注意本项目可以做bert seq2seq 任何任务，比如对联，写诗，自动摘要等等等等，只要你下载数据集，并且写好对应trainer.py，即可。如果喜欢的话欢迎star～ 如果遇到问题也可以提issue，保证会回复。

部分代码参考了 https://github.com/huggingface/transformers/ 和 https://github.com/bojone/bert4keras 
非常感谢！！！

### 运行
1. 下载想训练的数据集，可以专门建个corpus文件夹存放，参考config.py里面的一些配置。
2. 使用roberta模型，模型和字典文件需要去 https://drive.google.com/file/d/1iNeYFhCBJWeUsIlnW_2K6SMwXkM4gLb_/view 这里下载。 具体可以参考这个github仓库～ https://github.com/ymcui/Chinese-BERT-wwm
3. 如果使用普通的bert模型，下载bert中文预训练权重 "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin", 下载bert中文字典 "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt".
4. 去train文件夹下面运行对应的trainer.py，针对不同任务，运行不同trainer.py文件，进行训练。
5. 注意，代码里面目前都是使用roberta模型了，如果下载的普通bert模型，记得修改py文件里面的import部分。

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
