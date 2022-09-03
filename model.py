import transformers
from torch import nn
import torch
import argparse
from torch import optim
import pickle
from dataloader import create_dataloader


def train_model(dataloader,model,optimizer,device):
    model.train()
    loss_f = nn.CrossEntropyLoss()
    total_loss = 0
    correct_pred = 0
    for batch in dataloader:
        print(batch.shape)
        optimizer.zero_grad()
        instance = batch[0]
        batch_size = instance.shape[0]
        print(batch_size)
        mask = batch[1]
        label = batch[2]
        one_hot_label = nn.functional.one_hot(label,num_classes = 3)

        instance1 = instance[:,0,:].to(device)
        instance2 = instance[:,1,:].to(device)
        mask1 = mask[:,0,:].to(device)
        mask2 = mask[:,1,:].to(device)

        outputs = model(instance1,mask1,instance2,mask2)
        one_hot_label = one_hot_label.float().to(device)
        loss = loss_f(outputs, one_hot_label)
        total_loss += loss

        loss.backward()
        optimizer.step()

        correct_pred += calculate_correct_prediction(outputs,label)
    print(len(dataloader))
    print((len(dataloader)*batch_size))
    avg_loss = total_loss / (len(dataloader)*batch_size)
    avg_accuracy = correct_pred/(len(dataloader)*batch_size)
    #there is probably something wrong with the calculation of loss and accuracy... the accuracy got higher than 1.
    print(("-----------------Average Loss {}------------------".format(avg_loss)))
    print(("-----------------Average Accuracy {}------------------".format(avg_accuracy)))

def evaluate_model(dataloader,model,device):
    model.eval()
    loss_f = nn.CrossEntropyLoss()
    total_loss = 0
    correct_pred = 0

    with torch.no_grad():
        for batch in dataloader:
            instance = batch[0]
            batch_size = instance.shape[0]
            mask = batch[1]
            label = batch[2]
            one_hot_label = nn.functional.one_hot(label,num_classes = 3)

            instance1 = instance[:,0,:].to(device)
            instance2 = instance[:,1,:].to(device)
            mask1 = mask[:,0,:].to(device)
            mask2 = mask[:,1,:].to(device)

            outputs = model(instance1,mask1,instance2,mask2)
            one_hot_label = one_hot_label.float().to(device)
            loss = loss_f(outputs, one_hot_label)
            total_loss += loss
            correct_pred += calculate_correct_prediction(outputs,label)

        avg_loss = total_loss / (len(dataloader)*batch_size)
        avg_accuracy = correct_pred/(len(dataloader)*batch_size)
        print(("-----------------Average Loss {}------------------".format(avg_loss)))
        print(("-----------------Average Accuracy {}------------------".format(avg_accuracy)))

def train_and_evaluate(epoch,model,optimizer,train_dataloader,dev_dataloader,test_dataloader,device0,device1):

    for k in range(epoch):
        print(("-----------------Epoch {}------------------".format(k)))
        print("-----------------Training------------------")
        model.to(device0)
        train_model(train_dataloader, model, optimizer,device0)

        #save checkpoint
        model_path = str("/home/CE/zhangshi/mygithubprojects/csbert"+"model"+str(k)+".pt")
        torch.save(optimizer.state_dict(), model_path)
#        print("-----------------Evaluating------------------")
#        model.to(device1)
#        evaluate_model(dev_dataloader, model, device1)
#    print("-----------------Final Evaluation------------------")
#    evaluate_model(test_dataloader, model, device1)

    #add something here to save the model

def calculate_correct_prediction(outputs,label):
    predictions = torch.argmax(outputs, dim=1).tolist()
    label = label.tolist()
    n = 0
    for i,j in zip(predictions,label):
        if i==j:
            n+=1
    print("correct{}".format(n))
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
    parser.add_argument('--train',help = "path to train")
    parser.add_argument('--dev', help="path to dev")
    parser.add_argument('--test', help="path to test")
    args = parser.parse_args()

    with open(args.train, 'rb') as pickle_file:
        train_dataloader = pickle.load(pickle_file)
    with open(args.dev, 'rb') as pickle_file:
        dev_dataloader = pickle.load(pickle_file)
    with open(args.test, 'rb') as pickle_file:
        test_dataloader = pickle.load(pickle_file)

    epoch = 30
    model = CSBERT()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    device1 = torch.device('cuda:1')
    device0 = torch.device('cuda:0')

    train_and_evaluate(epoch,model,optimizer,train_dataloader,dev_dataloader,test_dataloader,device0,device1)
