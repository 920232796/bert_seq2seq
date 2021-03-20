import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")
import torch
from bert_seq2seq.tokenizer import  load_chinese_base_vocab, Tokenizer, T5PegasusTokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration

model_path = './state_dict/t5-chinese'
model = MT5ForConditionalGeneration.from_pretrained(model_path)
word2ix = load_chinese_base_vocab("./state_dict/t5-chinese/vocab.txt")
tokenizer = T5PegasusTokenizer(word2ix)
model.eval()
text = '从那之后，一发不可收拾。此后的近百年间，一共有十七位新娘在与君山一带失踪。有时十几年相安无事，有时短短一个月内失踪两名。一个恐怖传说迅速传开：与君山里住着一位鬼新郎，若是他看中了一位女子，便会在她出嫁的路上将她掳走，再把送亲的队伍吃掉。'
ids = tokenizer.encode(text)[0]
ids = torch.tensor(ids, dtype=torch.long).view(1, -1)

output = model.generate(ids,
                            decoder_start_token_id=word2ix["[CLS]"],
                            eos_token_id=word2ix["[SEP]"],
                            max_length=30).numpy()[0]

print(''.join(tokenizer.decode(output[ 1:])).replace(' ', ''))