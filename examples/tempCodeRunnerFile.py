# 测试一下自定义数据集
    vocab_path = "./state_dict/roberta_wwm_vocab.txt" # roberta模型字典的位置
    sents_src, sents_tgt = read_corpus("./corpus/Poetry", vocab_path)
    
    dataset = BertDataset(sents_src, sents_tgt, vocab_path)
    word2idx = load_chinese_base_vocab(vocab_path)
    tokenier = Tokenizer(word2idx)
    dataloader =  DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    for token_ids, token_type_ids, target_ids in dataloader:
        print(token_ids.shape)
        print(tokenier.decode(token_ids[0].tolist()))
        print(tokenier.decode(token_ids[1].tolist()))
        print(token_type_ids)
        print(target_ids.shape)
        print(tokenier.decode(target_ids[0].tolist()))
        print(tokenier.decode(target_ids[1].tolist()))
        break