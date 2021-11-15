## 训练文件说明

### roberta、bert
1. [roberta_THUCNews_auto_title.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/roberta_THUCNews_auto_title.py) 自动摘要任务，使用THUCNews数据集，数据量较大。
2. [roberta_auto_title_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/roberta_auto_title_train.py) 自动摘要任务，使用了另一个小的数据集。
3. [roberta_math_ques_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/roberta_math_ques_train.py) 自动解答小学数学题。
4. [relationship_classify_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/relationship_classify_train.py) 人物关系分类任务。
5. [roberta_semantic_matching_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/roberta_semantic_matching_train.py) 语义匹配任务。
6. [roberta_relation_extract_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/roberta_relation_extract_train.py) 三元组抽取任务。
7. [roberta_poem_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/roberta_poem_train.py) roberta模型自动写诗任务。
8. [roberta_participle_CRF_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/roberta_participle_CRF_train.py) 中文分词任务。
9. [roberta_medical_ner_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/roberta_medical_ner_train.py) NER任务，使用医学数据。
10. [roberta_couplets_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/roberta_couplets_train.py) roberta模型对联任务。
11. [roberta_news_classification_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/roberta_news_classification_train.py) 文本分类任务。
12. [roberta_coarsness_NER_CRF_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/roberta_coarsness_NER_CRF_train.py) 粗粒度NER任务，使用roberta+CRF。
13. [roberta_coarsness_NER_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/coarsness_NER_train.py) 粗粒度NER任务，使用roberta。
14. [roberta_fine_grained_NER_CRF_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/roberta_fine_grained_NER_CRF_train.py) 细粒度NER任务，使用Bert+CRF。
15. [roberta_large_auto_title_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/roberta_large_auto_title_train.py) roberta-large模型，自动标题任务。

### nezha
1. [nezha_auto_title_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/nezha_auto_title_train.py) nezha模型，自动摘要任务。
2. [nezha_relation_extract_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/nezha_relation_extract_train.py) nezha模型，关系抽取任务。
3. [nezha_auto_title_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/nezha_auto_title_train.py) 华为nezha模型，自动摘要任务。
4. [nezha_couplets_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/nezha_couplets_train.py) 华为nezha模型，自动对联任务

### T5
1. [t5_ancient_translation_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/t5_ancient_translation_train.py) t5模型进行古文翻译。
2. [t5_auto_title_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/nezha_relation_extract_train.py) t5模型，自动标题任务。

### GPT-2
1. [gpt2_generate_article.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/gpt2_generate_article.py) GPT-2自动生成文章任务。
2. [gpt2_explain_dream_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/gpt2_explain_dream_train.py) gpt模型，使用周公解梦数据集。
3. [gpt2_ancient_translation_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/gpt2_ancient_translation_train.py) gpt2模型进行古文翻译。
4. [gpt2_english_story_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/gpt2_english_story_train.py) gpt2模型自动生成英文故事。

### Simbert
1. [simbert_train.py](https://github.com/920232796/bert_seq2seq/blob/master/examples/simbert_train.py) SimBert模型生成相似句子。