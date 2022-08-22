import torch
import transformers
import json
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def load_data(data_path):
    train = []
    for line in open(data_path, 'rb'):
        train.append(json.loads(line))

    sentences = list()
    label = list()
    for i in train:
        if i["label"] == "entailment":
            label.append(1)
        elif i["label"] == "neutral":
            label.append(0)
        elif i["label"] == "contradiction":
            label.append(-1)

        if i["label"] != "-":
            sentences.append(i["sentence1"])
            sentences.append(i["sentence2"])

    if len(sentences)==2*len(label):
        return sentences, label
    else:
        raise Exception("Check your code.")

def create_dataloader(data_path,batch_size = 25):
    sentences,label = load_data(data_path)

    tokenizer = transformers.BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")
    tokens = tokenizer.batch_encode_plus(sentences, padding=True, return_token_type_ids=False)
    sentences = tokens['input_ids']
    masks = tokens['attention_mask']

    sentence_pairs = list()
    attention_mask_pairs = list()
    for i in range(len(sentences)):
        pair = list()
        if i%2==1 and i<len(sentences):
            pair.append(sentences[i])
            pair.append(sentences[i+1])
            sentence_pairs.append(pair)

    for i in range(len(sentences)):
        pair = list()
        if i%2==1 and i<len(sentences):
            pair.append(masks[i])
            pair.append(masks[i+1])
            attention_mask_pairs.append(pair)

    x = torch.tensor(sentence_pairs)
    x_mask = torch.tensor(attention_mask_pairs)
    y = torch.tensor(tokens.tolist())
    data = TensorDataset(x, x_mask, y)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, shuffle = True,batch_size=batch_size)

    return dataloader