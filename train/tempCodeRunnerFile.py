# 测试一下自定义数据集
    dataset = PoemDataset()
    word2idx = load_chinese_base_vocab()
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
