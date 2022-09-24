import torch
import transformers
import json
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import argparse
import pickle

def load_data(data_path):
    """
    :param data_path: path to the dataset
    :return: "sentences" is a list of sentences, "label" is a list of labels
    """
    data = []
    for line in open(data_path, 'rb'):
        try:
            data.append(json.loads(line))
        except:
            #there is a line in cmnli dataset that can't be loaded. I added this “try except” the avoid that line
            print("The following line cannot be loaded:")
            print(line)

    sentences = list()
    label = list()
    for i in data:
        label.append(i["label"])
        sentences.append(i["query"])
        sentences.append(i["title"])

    #each sentence pair has one label. If the number of sentences !=2* number of labels, it means there is something wrong
    if len(sentences)==2*len(label):
        return sentences, label
    else:
        raise Exception("Check your code.")

def create_dataloader(data_path,model_name ="hfl/chinese-bert-wwm", batch_size = 25):
    """
    :param data_path: path to the dataset to be loaded
    :param model_name: name of the model. Use 'hfl/chinese-macbert-base' for MacBert, 'hfl/chinese-bert-wwm' for bert-wwm
    :param batch_size: batch size. Default batch size is 25, because I am 25 years old
    :return: dataloader that can be used to train Chinese Sentence BERT
    """
    sentences,label = load_data(data_path)
    # sentences is a list like this:
    # [sentence 1 from sentence pair 1, sentence 2 from sentence pair 2, sentence 1 from sentence pair 2, sentence 2 from sentence pair 2,...]
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    tokens = tokenizer.batch_encode_plus(sentences, padding=True, return_token_type_ids=False)
    sentences = tokens['input_ids']
    masks = tokens['attention_mask']

    sentence_pairs = list()
    attention_mask_pairs = list()

    #this for loop organizes sentences into sentence pairs
    #sentence_pairs is a list like this: [[sentence 1 from sentence pair 1, sentence 2 from sentence pair 2], [sentence 1 from sentence pair 2, sentence 2 from sentence pair 2],...]
    for i in range(len(sentences)):
        pair = list()
        if i%2==0 and i<len(sentences):
            pair.append(sentences[i])
            pair.append(sentences[i+1])
            sentence_pairs.append(pair)

    # this for loop organizes attention masks into attention mask pairs
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