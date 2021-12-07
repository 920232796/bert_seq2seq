import torch
from bert_seq2seq.seq2seq_model import Seq2SeqModel
from bert_seq2seq.bert_cls_classifier import BertClsClassifier
from bert_seq2seq.bert_seq_labeling import BertSeqLabeling
from bert_seq2seq.bert_seq_labeling_crf import BertSeqLabelingCRF
from bert_seq2seq.bert_relation_extraction import BertRelationExtrac
from bert_seq2seq.bert_cls_multi_classifier import BertClsMultiClassifier
import torch.nn.functional as F
from bert_seq2seq.bert_cls_multi_seq2seq import ClsMultiSeq2SeqModel
from bert_seq2seq.simbert_model import SimBertModel
from bert_seq2seq.gpt2_generate_model import GPT2


def load_bert(word2ix, tokenizer=None, model_name="roberta", model_class="seq2seq", target_size=0, target=None):
    """
    model_path: 模型位置
    这是个统一的接口，用来加载模型的
    model_class : seq2seq or encoder
    """

    if model_class == "seq2seq":
        bert_model = Seq2SeqModel(word2ix, model_name=model_name, tokenizer=tokenizer)
        return bert_model
    elif model_class == "cls":
        if target_size == 0:
            raise Exception("必须传入参数 target_size，才能确定预测多少分类")
        bert_model = BertClsClassifier(word2ix, target_size, model_name=model_name)
        return bert_model
    elif model_class == "sequence_labeling":
        ## 序列标注模型
        if target_size == 0:
            raise Exception("必须传入参数 target_size，才能确定预测多少分类")
        bert_model = BertSeqLabeling(word2ix, target_size, model_name=model_name)
        return bert_model
    elif model_class == "sequence_labeling_crf":
        # 带有crf层的序列标注模型
        if target_size == 0:
            raise Exception("必须传入参数 target_size，才能确定预测多少分类")
        bert_model = BertSeqLabelingCRF(word2ix, target_size, model_name=model_name)
        return bert_model
    elif model_class == "relation_extrac":
        if target_size == 0:
            raise Exception("必须传入参数 target_size 表示预测predicate的种类")
        bert_model = BertRelationExtrac(word2ix, target_size, model_name=model_name)
        return bert_model
    elif model_class == "simbert":
        bert_model = SimBertModel(word2ix, model_name=model_name)
        return bert_model
    elif model_class == "multi_label_cls":
        bert_model = BertClsMultiClassifier(word2ix, target_size, model_name=model_name)
        return bert_model
    elif model_class == "multi_label_cls_seq2seq":
        bert_model = ClsMultiSeq2SeqModel(word2ix, target, model_name=model_name)
        return bert_model 
    else :
        raise Exception("model_name_err")


def load_gpt(word2ix, tokenizer=None):
    model = GPT2(word2ix, tokenizer=tokenizer)
    return model 


