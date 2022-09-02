import transformers
from torch import nn
import torch
import argparse
from torch import optim
import pickle
from dataloader import create_dataloader


def train_model(epoch,dataloader,model,optimizer):
    loss_f = nn.CrossEntropyLoss()
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
            one_hot_label = nn.functional.one_hot(label,num_classes = 3)

            instance1 = instance[:,0,:].to(device)
            instance2 = instance[:,1,:].to(device)
            mask1 = mask[:,0,:].to(device)
            mask2 = mask[:,1,:].to(device)

            outputs = model(instance1,mask1,instance2,mask2)
            outputs.float()
            one_hot_label.float()
            print(outputs.dtype)
            print(one_hot_label.dtype())
#            print(outputs.shape)
#            print(label.shape)
#            print(one_hot_label.shape)
#            torch.Size([25, 3])
#            torch.Size([25])
#            torch.Size([25, 3])
#            one_hot_label = one_hot_label.to(dtype=torch.long)
#            outputs = outputs.to(dtype=torch.long)
            loss = loss_f(outputs, one_hot_label)
            total_loss += loss

            loss.backward()
            optimizer.step()

            correct_pred += calculate_correct_prediction(outputs,label)

        avg_loss = total_loss / (len(dataloader)*len(dataloader[0]))
        avg_accuracy = correct_pred/(len(dataloader)*len(dataloader[0]))
        print(("-----------------Average Loss {}------------------".format(avg_loss)))
        print(("-----------------Average Accuracy {}------------------".format(avg_accuracy)))

def calculate_correct_prediction(outputs,label):
    predictions = torch.argmax(outputs, dim=1).tolist()
    label = label.tolist()
    n = 0
    for i,j in zip(predictions,label):
        if i==j:
            n+=1
    return n
class CSBERT(nn.Module):
    def __init__(self,model_name = "hfl/chinese-bert-wwm",pooling = "mean",freeze=0):
        super(CSBERT, self).__init__()

        self.bert = transformers.BertModel.from_pretrained(model_name)
#        if freeze !=0:
            #freeze bert layers here
#        self.pooling = nn.AvgPool1d(768, stride=768)
        self.linear1 = nn.Linear(768, out_features = 300)
        self.linear2 = nn.Linear(900, out_features = 3)
        self.softmax = nn.Softmax(dim=1)#I am not sure if this dimension is right... check later

    def forward(self,sent_id1,mask1,sent_id2,mask2):
        out1 = self.bert(sent_id1, attention_mask=mask1)
        pooled1 = torch.mean((out1[0] * mask1.unsqueeze(-1)), axis=1)#check if correct
        sentence_embedding1 = self.linear1(pooled1)

        out2 = self.bert(sent_id2, attention_mask=mask2)
        pooled2 = torch.mean((out2[0] * mask2.unsqueeze(-1)), axis=1)
        sentence_embedding2 = self.linear1(pooled2)

        embedding_concat = torch.cat((sentence_embedding1, sentence_embedding2, sentence_embedding1 - sentence_embedding2), 1)
        out_linear = self.linear2(embedding_concat)
        prediction = self.softmax(out_linear)

        return prediction



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the code')
    parser.add_argument('--path',help = "path to the pickled dataset")
    args = parser.parse_args()

    with open(args.path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    epoch = 10
    dataloader = data
    model = CSBERT()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_model(epoch, dataloader, model, optimizer)
