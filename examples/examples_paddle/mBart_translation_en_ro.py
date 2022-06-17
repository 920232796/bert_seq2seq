from paddlenlp.transformers import MBartForConditionalGeneration,MBartTokenizer
import paddle
from paddle.io import Dataset
import argparse
from tqdm import tqdm

parse = argparse.ArgumentParser()
parse.add_argument("--model_name",type=str)
parse.add_argument("--epoches",type=int)
parse.add_argument("--test_step",type=int)
parse.add_argument("--save_step",type=int)
parse.add_argument("--batch_size",type=int)
parse.add_argument("--src_lang",type=str)
parse.add_argument("--tgt_lang",type=str)
parse.add_argument("--datapath_en",type=str)
parse.add_argument("--datapath_ro",type=str)
parse.add_argument("--max_length",type=int)
opt = parse.parse_args()

def read_dataset(datapath_en,datapath_ro):
    dataset_en = []
    dataset_ro = []
    with open(datapath_en) as f:
        for line in f:
            dataset_en.append(line.strip("\n"))
    with open(datapath_ro) as ff:
        for line in ff:
            dataset_ro.append(line.strip("\n"))
    return dataset_en,dataset_ro

class EnRoDataset(Dataset):
    def __init__(self,dataset_en,dataset_ro):
        super(EnRoDataset,self).__init__()
        self.dataset_en = dataset_en
        self.dataset_ro = dataset_ro
    def __getitem__(self,index):
        data_for_en = self.dataset_en[index]
        data_for_ro = self.dataset_ro[index]
        input_ids = tokenizer.encode(data_for_en)["input_ids"]
        decoder_input_ids = [tokenizer.lang_code_to_id[opt.tgt_lang]]+tokenizer.encode(data_for_ro)["input_ids"][:-1]
        output = {
            "input_ids":input_ids,
            "decoder_input_ids":decoder_input_ids
        }
        return output
    def __len__(self):
        return len(self.dataset_en)
    @staticmethod
    def collate_fn(batch):
        def padding(indice,max_length=50,pad_idx=tokenizer.pad_token_id):
            pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
            return paddle.to_tensor(pad_indice)
        
        input_ids = [data["input_ids"] for data in batch]
        decoder_input_ids = [data["decoder_input_ids"] for data in batch]
        max_length_of_input_ids = max([len(text) for text in input_ids])
        max_length_of_decoder_input_ids = max([len(text) for text in decoder_input_ids])

        input_ids_padded = padding(input_ids,max_length_of_input_ids)
        decoder_input_ids_padded = padding(decoder_input_ids,max_length_of_decoder_input_ids)
        return input_ids_padded,decoder_input_ids_padded


model = MBartForConditionalGeneration.from_pretrained(opt.model_name)
tokenizer = MBartTokenizer.from_pretrained(opt.model_name,src_lang=opt.src_lang,tgt_lang=opt.tgt_lang)
dataset_en,dataset_ro = read_dataset(opt.datapath_en,opt.datapath_ro)
print(dataset_en[999])
print(dataset_ro[999])
dataset = EnRoDataset(
    dataset_en,
    dataset_ro)
dataloader = paddle.io.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    collate_fn=dataset.collate_fn)
optimizer = paddle.optimizer.AdamW(
    learning_rate=1e-5,
    parameters=model.parameters(),
    weight_decay=1e-5)

def calculate_loss(logits,label):
    return paddle.nn.functional.cross_entropy(logits.reshape([-1,tokenizer.vocab_size]),label.reshape([-1]))

def generate_text_for_test(text,target_language,max_length):
    with paddle.no_grad():
        input_ids = paddle.to_tensor(tokenizer.encode(text)["input_ids"]).unsqueeze(0)
        bos_id = tokenizer.lang_code_to_id[target_language]
        outputs, _ = model.generate(
            input_ids=input_ids,
            forced_bos_token_id=bos_id,
            decode_strategy="beam_search",
            num_beams=4,
            max_length=50)
    return tokenizer.convert_ids_to_string(outputs[0].numpy().tolist()[1:-1])

def train():
    
    global_step = 1
    report_loss = 0

    for epoch in range(opt.epoches):
        for input_ids, decoder_input_ids in tqdm(dataloader(), total=len(dataloader)):
            model.train()
            if global_step % opt.test_step == 0:
                model.eval()
                texts = ["election of Vice-Presidents of the European Parliament ( deadline for submitting nominations ) : see Minutes","agenda for next sitting : see Minutes"]
                for text in texts:
                    print("English:",text)
                    print("Romanian",generate_text_for_test(text,opt.tgt_lang,opt.max_length))
                print("loss is {}".format(report_loss))
                report_loss = 0
                model.train()
            if global_step % opt.save_step == 0:
                pass
            logits = model(input_ids=input_ids,decoder_input_ids=decoder_input_ids)
            loss = calculate_loss(logits[:,:-2],decoder_input_ids[:,1:-1])
            report_loss = report_loss + loss.item()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            global_step = global_step + 1
    


if __name__ == "__main__":
    train()

    # python new.py --model_name "mbart-large-en-ro" --epoches 3 --test_step 10 --save_step 10000 --batch_size 3 --src_lang "en_XX" --tgt_lang "ro_RO" --datapath_en "./wmt16_en_ro/corpus.en" --datapath_ro "./wmt16_en_ro/corpus.ro"  --max_length 128
    