# bert_seq2seq
一个轻量级的小框架。

pytorch实现bert做seq2seq任务，使用unilm方案。如果喜欢的话欢迎star～ 如果遇到问题也可以提issue，保证会回复。

也欢迎加入交流群～ 可以提问题，提建议，互相交流 QQ群: 975907202 

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

#### 医学ner 
##### 输入： 
如与其他药物同时使用可能会发生药物相互作用，详情请咨询医师或药师。  开水冲服，一次14克，一日3次。  养血，调经，止痛。用于月经量少、后错，经期腹痛  健民集团叶开泰国药(随州)有限公司  1，忌食生冷食物。2，患有其他疾病者，应在医师指导下服用。3，平素月经正常，突然出现月经过少，或经期错后，应去医院就诊。4，治疗痛经，宜在经前3～5天开始服药，连服一周，如有生育要求应在医师指导下服用。5，服药后痛经不减轻，或重度痛经者，应到医院诊治。6，服药2周症状无缓解，应去医院就诊。7，对本品过敏者禁用，过敏体质者慎用。8，本品性状发生改变时禁止使用。9，请将本品放在儿童不能接触的地方。10，如正在使用其他药品，使用本品前请咨询医师或药师。  本品为妇科月经不调类非处方药药品。  养血，调经，止痛。用于月经量少、后错，经期腹痛。 养血，调经，止痛。用于月经量少、后错，经期腹痛 14g*5袋  非处方药物（乙类）,国家医保目录（乙类）  孕妇禁用。糖尿病者禁服。

##### 输出：
![image.png](https://github.com/920232796/bert_seq2seq/blob/master/img/ner.jpg)
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
### 一些函数解释
#### def load_bert(word2ix, model_name="roberta", model_class="seq2seq")
加载bert模型，model_name参数指定了用哪种bert，目前支持bert和roberta；model_class指定了使用bert做哪种任务，seq2seq表示生成任务，cls表示文本分类任务......
#### def load_model_params(model, pretrain_model_path)
加载bert模型参数，注意，只是加载编码器的参数，也就是从网上下载好的预训练模型的参数；例如seq2seq模型包括了bert模型的参数+全连接层，此函数只是加载第一部分参数。
#### def load_recent_model(model, recent_model_path)
加载全部模型参数，当你训练了部分时间，保存了模型以后，通过此函数便可以加载上次模型训练结果，继续训练。

想看文章，可以去我网站～ http://www.blog.zhxing.online/#/  搜索写诗或者对联或者NER或者新闻摘要文本分类即可找到对应文章。
多谢支持。另外，网站上面还有一些介绍unilm论文和特殊的mask如何实现的文章，可以去网站里搜索一下。http://www.blog.zhxing.online/#/  搜索unilm 即可。

### 更新记录
2020.10.24: 调整了大量代码，添加了THUCNews数据集的自动摘要例子～现在的话，训练应该效果很好了，以前可能出现预训练参数加载不上的情况，效果有时会很差。

2020.10.23: 调整了一些代码结构，把每个例子里面的一些变量写为全局变量了，改了下beam-search的代码，更精简了。不过暂时不支持写诗里面的押韵了。以后补上。

2020.09.29: 新增了天池医学ner比赛的训练例子（医学ner_train.py），详情可见比赛界面：https://tianchi.aliyun.com/competition/entrance/531824/information

2020.08.16: 新增了诗词对联联合训练的例子(诗词对联_train.py)，可以同时写诗写词作对联了；另外新增了诗词的测试代码，模型训练好了可以进行测试。

2020.08.08: 本次更新的内容较多，1. 添加了自动摘要的例子(auto_title.py) 2. 添加了精简词表的代码，原本3W个字缩减为1W多（因为某些字永远不会出现） 3. 修改了部分beam-search代码，效果更好了。4. 细粒度ner暂时不能使用了，数据有点问题，因此暂时放入test文件夹，如果找到合适的数据，可以使用 5. 新增test文件夹，训练好的模型可以在里面进行测试，看看效果。

2020.06.22: 补充了Conditional Layer Norm 的一篇文章。解释了部分代码。http://www.blog.zhxing.online/#/readBlog?blogId=347

2020.06.21: 更新了很多代码，复现了一个三元组抽取的例子(三元组抽取_train.py)～

2020.06.02: 最近一直在忙毕业的事情，还有个比赛，暂时不更新了，以后会一直更新哒。

2020.04.18: 训练了bert+crf模型，crf层学习率好像不够高，还需要改善(现在已经可以单独设置crf层学习率了，一般设为0.01)。

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
