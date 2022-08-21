import pandas
import torch
import transformers
import json
#from sentence_transformers import SentenceTransformer, models, losses

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
        #add a shuffle here
        return sentences, label
    else:
        raise Exception("Check your code.")

def create_dataloader(sentences,label):
    tokenizer = transformers.BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")
    sent_id = tokenizer.batch_encode_plus(sentences, padding=True, return_token_type_ids=False)

def train_model():
    pass
def evaluate_model():
    pass
def create_model():
    word_model = transformers.BertModel.from_pretrained("hfl/chinese-bert-wwm")


    


#pooling_model
#sentence model is word model and pooling model ensembled together
#dataloader is always the same as in the transformers module
#train_loss = losses.CosineSimilarityLoss(model)