
import torch
from bert_seq2seq.model.bart_model import BartConfig, BartForConditionalGeneration, BartModel, shift_tokens_right
from bert_seq2seq.tokenizer import Tokenizer,load_chinese_base_vocab
from bert_seq2seq.basic_bert import BasicBart
from bert_seq2seq.seq2seq_model import top_k_top_p_filtering
import torch.nn.functional as F
import torch.nn as nn 

class BartGenerationModel(BasicBart):

    def __init__(self, word2idx):
        super().__init__()
        config = BartConfig()
        self.config = config 
        self.model = BartModel(config)
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.word2idx = word2idx
        self.tokenizer = Tokenizer(self.word2idx)
        self.bos_id = self.word2idx["[CLS]"]
        self.eos_id = self.word2idx["[SEP]"]
        self.unk_id = self.word2idx["[UNK]"]

    def forward(self, input_ids, decoder_input_ids, labels=None):
        input_ids = input_ids.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        decoder_out, _ = self.model(
            input_ids,
            decoder_input_ids=decoder_input_ids,
        )

        lm_logits = self.lm_head(decoder_out)
        target_mask = (decoder_input_ids > 0).float().view(-1)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = (loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1)) * target_mask).sum() / target_mask.sum()

        output = (lm_logits,)
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output


    def sample_generate_encoder_decoder(self, text, input_max_length=256, out_max_length=200, top_k=30, top_p=0.0, add_eos=True):

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
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if self.eos_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                input_decoder_ids = torch.cat((input_decoder_ids, next_token.long().unsqueeze(0)), dim=1)

        return self.tokenizer.decode(output_ids)