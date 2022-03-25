
import torch
from bert_seq2seq.model.t5_model import T5ForConditionalGeneration, T5Config, T5SmallConfig
from bert_seq2seq.tokenizer import T5PegasusTokenizer,load_chinese_base_vocab
from bert_seq2seq.basic_bert import BasicT5
from bert_seq2seq.seq2seq_model import top_k_top_p_filtering
import torch.nn.functional as F

class T5Model(BasicT5):

    def __init__(self, word2idx, size="base"):
        super().__init__()
        if size == "base":
            config = T5Config()
        elif size == "small":
            config = T5SmallConfig()
        else:
            raise Exception("not support this model type")
        self.model = T5ForConditionalGeneration(config)

        self.word2idx = word2idx
        self.tokenizer = T5PegasusTokenizer(self.word2idx)
        self.bos_id = self.word2idx["[CLS]"]
        self.eos_id = self.word2idx["[SEP]"]
        self.unk_id = self.word2idx["[UNK]"]

    def forward(self, input_ids, decoder_input_ids, labels=None):
        input_ids = input_ids.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        return self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)


    def sample_generate_encoder_decoder(self, text, input_max_length=256, out_max_length=200, top_k=30, top_p=1.0, add_eos=True):
        
        token_out = self.tokenizer.encode(text, max_length=input_max_length)
        if len(token_out) == 2:
            token_ids = token_out[0]
        else:
            token_ids = token_out
        if not add_eos:
            token_ids = token_ids[:-1]
        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(1, -1)
        output_ids = []

        input_decoder_ids = torch.tensor(self.bos_id, device=self.device, dtype=torch.long).view(1, -1)
        with torch.no_grad():
            for step in range(out_max_length):
                scores = self.model(input_ids=token_ids, decoder_input_ids=input_decoder_ids)[0]
                logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
                logit_score[self.unk_id] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)

                filterd_logits_prob = F.softmax(filtered_logits, dim=-1)
                
                next_token = torch.multinomial(filterd_logits_prob, num_samples=1)
                if self.eos_id == next_token.item():
                    break
                
                output_ids.append(next_token.item())
                input_decoder_ids = torch.cat((input_decoder_ids, next_token.long().unsqueeze(0)), dim=1)

        return self.tokenizer.decode(output_ids)