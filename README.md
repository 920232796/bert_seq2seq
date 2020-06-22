# bert_seq2seq
一个轻量级的小框架。

pytorch实现bert做seq2seq任务，使用unilm方案。如果喜欢的话欢迎star～ 如果遇到问题也可以提issue，保证会回复。

本框架目前可以做各种NLP任务，一共分为四种：
1. seq2seq 比如写诗，对联，自动摘要等。
2. cls_classifier 通过提取句首的cls向量去做分类，比如情感分析，文本分类。
3. sequence_labeling 序列标注任务，比如命名实体识别，词性标注。
4. relation_extract 关系抽取，比如三元组抽取任务。(复现苏剑林老师的例子，不完全一样。)
四种任务分别加载四种不同的模型，通过``` model_class="seq2seq" or "cls" or "sequence_labeling" or "sequence_labeling_crf or relation_extrac"``` 参数去设置。具体可以去看examples里面的各种例子。当然也随时可以查看修改我的源代码～

部分代码参考了 https://github.com/huggingface/transformers/ 和 https://github.com/bojone/bert4keras 
非常感谢！！！

### 目前几个小例子的效果截图
#### 写诗
![image.png](http://www.zhxing.online/image/acb592f918894ca6b62435d2464d3cb0.png)
#### 新闻摘要文本分类（14分类）
![image.png](http://www.zhxing.online/image/724f93b03c19404fba4f684eac4695bc.png)
输出：
![image.png](http://www.zhxing.online/image/4175b02f928f43fc84e9c866aba3ee2d.png)
#### 对联
![image.png](http://www.zhxing.online/image/42eec322d6cc419da0efdc45c02d9f25.png)

### 安装 
1. 安装本框架 ```pip install bert-seq2seq```
2. 安装pytorch 
3. 安装tqdm 可以用来显示进度条 ```pip install tqdm```
### 运行
1. 下载想训练的数据集，可以专门建个corpus文件夹存放。
2. 使用roberta模型，模型和字典文件需要去 https://drive.google.com/file/d/1iNeYFhCBJWeUsIlnW_2K6SMwXkM4gLb_/view 这里下载。 具体可以参考这个github仓库～ https://github.com/ymcui/Chinese-BERT-wwm
3. 如果使用普通的bert模型，下载bert中文预训练权重 "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin", 下载bert中文字典 "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt".
4. 去example文件夹下面运行对应的trainer.py，针对不同任务，运行不同train.py文件，需要修改输入输出数据的结构，然后进行训练。具体可以看examples里面的各种例子～

想看文章，可以去我网站～ http://www.blog.zhxing.online/#/  搜索写诗或者对联或者NER或者新闻摘要文本分类即可找到对应文章。
多谢支持。另外，网站上面还有一些介绍unilm论文和特殊的mask如何实现的文章，可以去网站里搜索一下。http://www.blog.zhxing.online/#/  搜索unilm 即可。

### 更新记录

2020.06.22: 补充了Conditional Layer Norm 的一篇文章。解释了部分代码。http://www.blog.zhxing.online/#/readBlog?blogId=347

2020.06.21: 更新了很多代码，复现了一个三元组抽取的例子～

2020.06.02: 最近一直在忙毕业的事情，还有个比赛，暂时不更新了，以后会一直更新哒。

2020.04.18: 训练了bert+crf模型，crf层学习率好像不够高，还需要改善。

2020.04.13: 添加了NER任务 + CRF层Loss，跑通了训练例子，但是还没有添加维特比算法。

2020.04.11: 计划给NER任务添加一个CRF层。

2020.04.07: 添加了一个ner的example。

2020.04.07: 更新了pypi，并且加入了ner等序列标注任务的模型。

2020.04.04: 更新了pypi上面的代码，目前最新版本 0.0.6，请用最新版本，bug会比较少。

2020.04.04: 修复了部分bug，添加了新闻标题文本分类的例子

2020.04.02: 修改了beam-search中对于写诗的重复字和押韵惩罚程度，可能效果会更好。

2020.04.02: 添加了周公解梦的task

2020.04.02: 添加了对对联的task

2020.04.01: 添加了写诗的task

2020.04.01: 重构了代码，开始训练一个新的任务花费时间更少。

python setup.py sdist
twine upload dist/bert_seq2seq-0.0.8.tar.gz

