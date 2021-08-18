import torch
from bert_seq2seq.tokenizer import  load_chinese_base_vocab, T5PegasusTokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from bert_seq2seq.extend_model_method import ExtendModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transformer t5 代码
model_path = './state_dict/t5-chinese'
model = MT5ForConditionalGeneration.from_pretrained(model_path)
word2ix = load_chinese_base_vocab("./state_dict/t5-chinese/vocab.txt")
tokenizer = T5PegasusTokenizer(word2ix)
model.eval()
model = ExtendModel(model, tokenizer=tokenizer, bos_id=word2ix["[CLS]"], eos_id=word2ix["[SEP]"], device=device)
text = '从那之后，一发不可收拾。此后的近百年间，一共有十七位新娘在与君山一带失踪。有时十几年相安无事，有时短短一个月内失踪两名。一个恐怖传说迅速传开：与君山里住着一位鬼新郎，若是他看中了一位女子，便会在她出嫁的路上将她掳走，再把送亲的队伍吃掉。'
out = model.sample_generate_encoder_decoder(text)
print(out)

# 加载自己t5代码
from bert_seq2seq.t5_ch import T5Model
vocab_path = "./state_dict/t5-chinese/vocab.txt"
model = T5Model(vocab_path, size="base")
model.set_device(device)
model.load_pretrain_params("./state_dict/t5-chinese/pytorch_model.bin")
model.eval()
text = '从那之后，一发不可收拾。此后的近百年间，一共有十七位新娘在与君山一带失踪。有时十几年相安无事，有时短短一个月内失踪两名。一个恐怖传说迅速传开：与君山里住着一位鬼新郎，若是他看中了一位女子，便会在她出嫁的路上将她掳走，再把送亲的队伍吃掉。'
out = model.sample_generate_encoder_decoder(text)
print(out)


