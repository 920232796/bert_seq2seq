# bert_seq2seq
一个轻量级的小框架，如果喜欢的话欢迎star～ 谢谢谢谢。如果遇到问题也可以提issue，保证会回复。

## 欢迎加入交流群～ 可以提问题，提建议，互相交流 QQ群: 975907202

### 本框架目前可以做各种NLP任务，支持的模型有：
1. bert
2. roberta 
3. roberta-large 
4. gpt2
5. t5
6. 华为nezha模型
7. bart-中文

### 支持的任务有：
1. seq2seq 比如写诗，对联，自动标题，自动摘要等。
2. cls_classifier 通过提取句首的cls向量去做分类，比如情感分析，文本分类，语义匹配等。
3. sequence_labeling 序列标注任务，比如命名实体识别，词性标注，中文分词等。
4. sequence_labeling_crf 加入CRF Loss的序列标注任务，效果更好。
4. relation_extract 关系抽取，比如三元组抽取任务。(复现苏剑林老师的例子，不完全一样。)
5. simbert SimBert模型，生成相似句子，并且给相似句子进行相似度判断。
6. multi_label_cls 多标签分类。

加载不同模型通过设置“model_name”参数实现，不同任务通过设置“model_class”参数实现，具体可以去看examples里面的各种例子。
### 预训练模型下载地址汇总：
1. roberta模型，模型和字典文件需要去 https://drive.google.com/file/d/1iNeYFhCBJWeUsIlnW_2K6SMwXkM4gLb_/view 这里下载。 具体可以参考这个github仓库～ https://github.com/ymcui/Chinese-BERT-wwm ，roberta-large模型也是在里面进行下载即可。
2. bert模型(目前不支持large)，下载bert中文预训练权重 "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin", 下载bert中文字典 "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt".
3. nezha模型，字典权重位置（目前只支持base）：nezha-base模型下载：链接: https://pan.baidu.com/s/1Z0SJbISsKzAgs0lT9hFyZQ 提取码: 4awe
4. gpt2模型，可以查看test文件中的gpt_test文件进行文本续写测试，gpt2中文通用模型和字典下载地址：https://pan.baidu.com/s/1vTYc8fJUmlQrre5p0JRelw  密码: f5un
5. gpt2英文模型，具体参考了https://huggingface.co/pranavpsv/gpt2-genre-story-generator 这个预训练的model，具体的训练代码可以看example中的gpt2_english_story_train.py
6. 支持t5模型，英文中文都支持，直接使用transformers包进行加载，具体可以看examples文件夹中的相关例子。 预训练参数下载：https://github.com/renmada/t5-pegasus-pytorch
7. SimBert模型，支持相似句的生成，预训练模型使用bert、roberta、nezha均可。
8. bart中文模型下载地址：https://huggingface.co/fnlp/bart-base-chinese

部分代码参考了 https://github.com/huggingface/transformers/ 和 https://github.com/bojone/bert4keras 非常感谢！！！

### 一些小例子的效果截图

#### gpt2生成
##### 输入：
今天天气好
##### 输出：
，就 和 宝 贝 们 一 起 去 那 里 看 电 影 了 ，真 的 好 好 哦 ！环 境 什 么 的 都 是 没 得 说 的 ，电 影 很 精 致 ，音 效 也 很 不 错 ，不 知 道 这 家 店 还 开 着 没 有 ，希 望 有 空 的 话 可 以 经 常 去 看 看
#### 写诗
![image.png](http://www.zhxing.online/image/acb592f918894ca6b62435d2464d3cb0.png)
#### bert+crf ner
输入：
![image.png](img/ner-input.png?raw=true)
输出：
![iamge.png](https://github.com/920232796/bert_seq2seq/raw/master/img/ner-out.png)
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

#### 语义匹配
![image.png](https://blog-image-xzh.oss-cn-beijing.aliyuncs.com/c83d27eb-3c2d-4a5c-9496-cd635a0094be.jpg)

#### 分词
![image.png](/img/fenci.png)

### 安装 
1. 安装本框架 ```pip install bert-seq2seq```
2. 安装pytorch 
3. 安装tqdm 可以用来显示进度条 ```pip install tqdm```
4. 准备好自己的数据，只需要修改example代码中的read_data函数，构造好输入输出，便可开始训练。
5. 去example文件夹下面运行对应的*_train.py文件，针对不同任务，运行不同train.py文件，需要修改输入输出数据的结构，然后进行训练。具体可以看examples里面的各种例子～

### 一些函数解释
#### def load_bert(word2ix, model_name="roberta", model_class="seq2seq")
加载bert模型，model_name参数指定了用哪种bert，目前支持bert、roberta、nezha；model_class指定了使用bert做哪种任务，seq2seq表示生成任务，cls表示文本分类任务......
#### model.load_pretrain_params(pretrain_model_path)
加载bert模型参数，注意，只是加载编码器的参数，也就是从网上下载好的预训练模型的参数；例如seq2seq模型包括了bert模型的参数+全连接层，此函数只是加载第一部分参数。
#### def model.load_all_params(recent_model_path)
加载全部模型参数，当你训练了部分时间，保存了模型以后，通过此函数便可以加载上次模型训练结果，继续训练或者进行测试。

想看各种文章，可以去我网站～ http://www.blog.zhxing.online/#/  搜索写诗或者对联或者NER或者新闻摘要文本分类即可找到对应文章。
多谢支持。

### 更新记录

2021.11.12: 优化代码，支持了roberta-large模型。

2021.10.12: 优化了ner的解码方法，以前粗粒度的解码方式存在bug。

2021.08.18: 优化了大量代码，目前框架代码看起来更加清晰了，删除了大量冗余的代码。

2021.08.17: 支持了华为的nezha模型，很简单，改一下model_name参数即可，欢迎测试效果。

2021.08.15: 添加了分词的例子，tokenizer中添加了rematch代码。

2021.07.29： 优化部分代码，更简洁了。

2021.07.20: 复现了SimBert模型，可以进行相似句的输出，不过由于数据量太少，还有待测试。

2021.03.19: 支持模型扩展，可以不仅仅使用框架自带的模型了，可以直接加载hugging face上面的模型进行训练 预测。

2021.03.12: 添加了gpt2中文训练的例子，周公解梦。

2021.03.11: 添加了gpt2例子，可以进行文章的续写。

2021.03.11: 添加了一个随机生成的解码方式，生成更加多样了。

2021.03.08: beam search 返回n个结果，随机取某个作为输出。

2021.02.25: 添加了一个语义匹配的例子。

2021.02.06: 调整了device的设置方式，现在更加的方便了。

2021.1.27: 调整了框架的代码结构，改动较多，如果有bug，欢迎提issue。

2021.1.21: 添加了一个新的例子，人物关系提取分类。

2020.12.02: 调整了一些代码，并且添加了几个测试的文件，可以很方便的加载已经训练好的模型，进行对应任务的测试。

2020.11.20: 添加了一个例子，三元组抽取f1目前能到0.7。添加了新闻摘要文本分类的测试代码。

2020.11.04: 跑了跑bert-crf做普通ner任务的例子，效果不错。

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
twine upload dist/bert_seq2seq-2.3.4.tar.gz
