import pandas
import torch
import transformers
import json
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import dataloader
#from sentence_transformers import CosineSimilarityLoss
#from sentence_transformers import SentenceTransformer, models, losses

def cosione_similarity_loss(embeddings,labels):
    pass

def train_model(epoch,dataloader,model,optimizer):
    loss_f = CosineSimilarityLoss(model)
    total_loss = 0
    batch_size = len(epoch[0])
    for i in epoch:
        print("-----------------Epoch {}------------------".format(i))

        for batch in dataloader:
            sent_id, mask, labels = batch
            sen_embeds = model(sent_id,mask)
            loss = loss_f(sen_embeds,labels)
            total_loss += loss
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(train_dataloader)
        print(("-----------------Average Loss {}------------------".format(avg_loss)))


def evaluate_model(dataloader):
    total = len(dataloader)
    for instance in dataloader:
        sent_id, mask, labels = instance


class CosineSimilarityLoss(nn.Module):

    def __init__(self, model, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation


    def forward(self, sentence_features, labels):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        return self.loss_fct(output, labels.view(-1))


class CSBERT(nn.Module):

    def __int__(self,model_name = "hfl/chinese-bert-wwm",pooling = "mean",out_features = 265,freeze=0):
        super(CSBERT, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(model_name)
        if freeze !=0:
            #freeze bert layers here
            pass
        self.pooling = self.pooling_layer(pooling)
        self.linear = nn.Linear(123, out_features = out_features)

    def forward(self,sent_id,mask):
        # pass the inputs to the model
        outputs = self.bert(sent_id, attention_mask=mask)#not sure if the output is correct. Needs to be checked
        pooled = self.pooling(outputs.last_hidden_state)#need to be changed
        sentence_embedding = self.linear(pooled)

        return sentence_embedding


    def pooling_layer(self,pooling,feature):
        assert pooling in ["mean","cls"]
        pooled = list()
        if pooling == "cls":
            pooled.append(feature[:, 0])#This could be wrong. Check this later
        else:
            pass

        return pooled
    def get_word_embedding_dimension(self) -> int:
        pass

#pooling_model
#sentence model is word model and pooling model ensembled together
#dataloader is always the same as in the transformers module
#train_loss = losses.CosineSimilarityLoss(model)