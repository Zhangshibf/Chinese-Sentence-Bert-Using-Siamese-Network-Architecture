import torch
import transformers
import json
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import argparse
import pickle

def load_data(data_path):
    data = []
    for line in open(data_path, 'rb'):
        try:
            data.append(json.loads(line))
        except:
            print(line)

    sentences = list()
    label = list()
    for i in data:
        if i["label"] != "-": #some sentence pairs have no label. These instances are excluded from train set
            if i["label"] == "entailment":
                label.append(0)
            elif i["label"] == "neutral":
                label.append(1)
            elif i["label"] == "contradiction":
                label.append(2)

            sentences.append(i["sentence1"])
            sentences.append(i["sentence2"])

    if len(sentences)==2*len(label):
        return sentences, label
    else:
        raise Exception("Check your code.")

def create_dataloader(data_path,model_name ="hfl/chinese-bert-wwm", batch_size = 25):
    sentences,label = load_data(data_path)

    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    tokens = tokenizer.batch_encode_plus(sentences, padding=True, return_token_type_ids=False)
    sentences = tokens['input_ids']
    masks = tokens['attention_mask']

    sentence_pairs = list()
    attention_mask_pairs = list()
    for i in range(len(sentences)):
        pair = list()
        if i%2==0 and i<len(sentences):
            pair.append(sentences[i])
            pair.append(sentences[i+1])
            sentence_pairs.append(pair)


    for i in range(len(sentences)):
        pair = list()
        if i%2==0 and i<len(sentences):
            pair.append(masks[i])
            pair.append(masks[i+1])
            attention_mask_pairs.append(pair)

    x = torch.tensor(sentence_pairs)
    x_mask = torch.tensor(attention_mask_pairs)
    y = torch.tensor(label)
    data = TensorDataset(x, x_mask, y)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create the dataloader and pickle it.')
    parser.add_argument('--i',help = "path to the dataset", required=True)
    parser.add_argument('--o',help = "path to save the pickled dataloader", required=True)
    parser.add_argument('--model_name', help="use 'hfl/chinese-macbert-base' for MacBert, 'hfl/chinese-bert-wwm' for bert-wwm. Default bert-wwm",
                        default="hfl/chinese-bert-wwm")
    args = parser.parse_args()
    dataloader = create_dataloader(args.i,args.model_name)

    with open(args.o, 'wb') as f:
        pickle.dump(dataloader, f)