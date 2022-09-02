import transformers
from torch import nn
import torch
import argparse
from torch import optim
import pickle
from dataloader import create_dataloader


def train_model(epoch,dataloader,model,optimizer):
    loss_f = nn.CosineEmbeddingLoss
    total_loss = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    correct_pred = 0
    for k in range(epoch):
        print("-----------------Training Epoch {}------------------".format(k+1))

        for batch in dataloader:
            optimizer.zero_grad()
            instance = batch[0]
            mask = batch[1]
            label = batch[2]

            instance1 = instance[:,0,:].to(device)
            instance2 = instance[:,1,:].to(device)
            mask1 = mask[:,0,:].to(device)
            mask2 = mask[:,1,:].to(device)

            outputs = model(instance1,mask1,instance2,mask2)
            loss = loss_f(outputs, label)
            total_loss += loss

            loss.backward()
            optimizer.step()
            #update correct prediction
        avg_loss = total_loss / (len(dataloader)*len(dataloader[0]))
        avg_accuracy = correct_pred/(len(dataloader)*len(dataloader[0]))
        print(("-----------------Average Loss {}------------------".format(avg_loss)))
        print(("-----------------Average Accuracy {}------------------".format(avg_loss)))


class CSBERT(nn.Module):
    def __init__(self,model_name = "hfl/chinese-bert-wwm",pooling = "mean",freeze=0):
        super(CSBERT, self).__init__()

        self.bert = transformers.BertModel.from_pretrained(model_name)
#        if freeze !=0:
            #freeze bert layers here
        self.pooling = nn.AvgPool1d(768, stride=768)#need to be changed
        self.linear = nn.Linear(768, out_features = 3)
        self.softmax = nn.Softmax(dim=1)#I am not sure if this dimension is right... check later

    def forward(self,sent_id1,mask1,sent_id2,mask2):
        out = self.bert(sent_id1, attention_mask=mask1)
        print(out[0].shape)
        pooled = out[0].sum(axis=1)/62
        print(pooled.shape)
        averaged = torch.mean((out[0] * mask1.unsqueeze(-1)), axis=1)
        print(averaged.shape)
#        pooled1 = self.pooling(out[0].permute(0,2,1)).permute(0,2,1)
        print(pooled1.shape)
        #use attention mask for pooling. Don't pool tokens that are padding
        #pooled 后size应该是25,768
        sentence_embedding1 = self.linear(pooled1)

        out = self.bert(sent_id2, attention_mask=mask2)
        pooled2 = self.pooling(out[0])
        sentence_embedding2 = self.linear(pooled2)

        embedding_concat = torch.cat((sentence_embedding1, sentence_embedding2, sentence_embedding1 - sentence_embedding2), 0)
        prediction = self.softmax(embedding_concat)

        return prediction



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the code')
    parser.add_argument('--path',help = "path to the pickled dataset")
    args = parser.parse_args()

    with open(args.path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    print("Get ready for error!")
    epoch = 10
    dataloader = data
    model = CSBERT()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_model(epoch, dataloader, model, optimizer)
