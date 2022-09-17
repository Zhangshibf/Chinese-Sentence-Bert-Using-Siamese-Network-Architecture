import transformers
from torch import nn
import torch
import argparse
from torch import optim
import pickle
import numpy as np
from scipy import stats



def train_model(k,dataloader,model,optimizer,device,save_model,output_path):
    model.train()
    loss_f = nn.CrossEntropyLoss()
    total_loss = 0
    correct_pred = 0
    total_num = 0
    for batch in dataloader:
        total_num+=len(batch[0])
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
        one_hot_label = one_hot_label.float().to(device)
        loss = loss_f(outputs, one_hot_label)
        total_loss += loss

        loss.backward()
        optimizer.step()

        correct = calculate_correct_prediction(outputs,label)
        correct_pred+=correct

    avg_loss = total_loss / total_num
    avg_accuracy = correct_pred/total_num

    print(("-----------------Average Loss {}------------------".format(avg_loss)))
    print(("-----------------Average Accuracy {}------------------".format(avg_accuracy)))
#"/home/CE/zhangshi/mygithubprojects/csbert_macbert/"
    if save_model==True:
        model_path = str(str(output_path) + "macbertmodel" + str(k) + ".pt")
        torch.save(model.state_dict(), model_path)
        print("Model saved, path is {}".format(model_path))

    return avg_loss,avg_accuracy

def evaluate_model(dataloader,model,device):
    model.eval()
    loss_f = nn.CrossEntropyLoss()
    total_loss = 0
    correct_pred = 0
    total_num = 0

    with torch.no_grad():
        for batch in dataloader:
            total_num += len(batch[0])
            instance = batch[0]
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

    avg_loss = total_loss / total_num
    avg_accuracy = correct_pred/total_num
    print(("-----------------Average Loss {}------------------".format(avg_loss)))
    print(("-----------------Average Accuracy {}------------------".format(avg_accuracy)))

    return avg_loss, avg_accuracy


def evaluate_model_cosine_similarity(dataloader,model,device):
    #return pearson's correlation
    model.eval()
    total_num = 0
    similarity_scores = list()
    labels = list()
    with torch.no_grad():
        for batch in dataloader:
            total_num += len(batch[0])
            instance = batch[0]
            mask = batch[1]
            label = batch[2]
            label = label.tolist()
            """
                        if i["label"] == "entailment":
                label.append(0)
            elif i["label"] == "neutral":
                label.append(1)
            elif i["label"] == "contradiction":
                label.append(2)"""
            for i in label:
                if i ==0:
                    labels.append(1)
                elif i ==1:
                    labels.append(0)
                elif i ==2:
                    labels.append(-1)
            instance1 = instance[:,0,:].to(device)
            instance2 = instance[:,1,:].to(device)
            mask1 = mask[:,0,:].to(device)
            mask2 = mask[:,1,:].to(device)

            embedding1 = model.generate_sentence_embedding(instance1,mask1)
            embedding2 = model.generate_sentence_embedding(instance2,mask2)
            similarity = nn.functional.cosine_similarity(embedding1,embedding2)
            similarity_scores.extend(similarity.tolist())

    #calculate spearman's r
    spear = stats.spearmanr(similarity_scores,labels)
    print(spear)
    r = spear[0]

    print(("---------Spearman's rank coefficient is {}---------".format(r)))
    return r

def train_and_save_model(epoch,model,optimizer,train_dataloader,device,output_path):
    loss_list = list()
    accuracy_list = list()
    for k in range(epoch):
        print(("-----------------Epoch {}------------------".format(k)))
        print("-----------------Training------------------")
        model.to(device)
        loss, acc = train_model(k,train_dataloader, model, optimizer,device,save_model=True,output_path=output_path)
        loss = str(loss.tolist())
        acc = str(acc)
        loss_list.append(loss)
        accuracy_list.append(acc)
        result_path = str(output_path + "train_result.txt")
        with open(result_path, "a") as f:
            f.write(str("Epoch "+str(k)))
            f.write(str("loss: "+loss + "\n"))
            f.write("accuracy: "+acc + "\n")
            f.close()

    print(loss_list)
    print(accuracy_list)

def evaluate_saved_model(epoch,model_name,model_path,dev_dataloader,device,outpath):
    spearman_r = list()
    for k in range(int(epoch)):
        print(("-----------------Model Saved at Epoch {}------------------".format(k)))
        print("-----------------Evaluating------------------")
        model = CSBERT(model_name)
        path = str(model_path+"model"+str(k)+".pt")
        model.load_state_dict(torch.load(path))
        model.to(device)
        r = evaluate_model_cosine_similarity(dev_dataloader, model, device)
        spearman_r.append(str(r))
        rs = " ".join(spearman_r)
        with open(str(outpath+"dev_result.txt"), "a") as f:
            f.write((str(k)+"\n"))
            f.write((rs+"\n"))
            f.close()

    #find max of accuracy list
    ind = np.argmax(np.array(spearman_r))
    print(("-----------------Best Performance at Epoch {}------------------".format(ind)))
    best_performance_model_path = str(model_path+"model"+str(ind)+".pt")

    return best_performance_model_path



def calculate_correct_prediction(outputs,label):
    predictions = torch.argmax(outputs, dim=1).tolist()
    label = label.tolist()
    n = 0
    for i,j in zip(predictions,label):
        if i==j:
            n+=1
    return n
class CSBERT(nn.Module):
    def __init__(self,model_name ,pooling = "mean",freeze=0):
        super(CSBERT, self).__init__()

        self.bert = transformers.BertModel.from_pretrained(model_name)
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

    def generate_sentence_embedding(self,sent_id,mask):
        out = self.bert(sent_id, attention_mask=mask)
        pooled = torch.mean((out[0] * mask.unsqueeze(-1)), axis=1)  # check if correct
        sentence_embedding = self.linear1(pooled)

        return sentence_embedding


"""
def evaluate_saved_model(epoch,model_path,dev_dataloader,device):
    loss_list = list()
    accuracy_list = list()
    for k in range(int(epoch)):
        print(("-----------------Model Saved at Epoch {}------------------".format(k)))
        print("-----------------Evaluating------------------")
        model = CSBERT()
        path = str(model_path+"model"+str(k)+".pt")
        model.load_state_dict(torch.load(path))
        model.to(device)
        loss, acc = evaluate_model(dev_dataloader, model, device)
        loss_list.append(str(loss.tolist()))
        accuracy_list.append(str(acc))

    print(loss_list)
    print(accuracy_list)

    with open("/home/CE/zhangshi/mygithubprojects/csbert_macbert/dev_result.txt", "a") as f:
        l = " ".join(loss_list)
        a = " ".join(accuracy_list)
        f.write(str(l+"\n"))
        f.write(str(a+"\n"))
        f.close()

    #find max of accuracy list
    ind = np.argmax(accuracy_list)
    print(("-----------------Best Performance at Epoch {}------------------".format(ind)))
    best_performance_model_path = str(model_path+"model"+str(ind)+".pt")

    return best_performance_model_path
"""

"""
class CSBERT(nn.Module):
    def __init__(self,model_name ,pooling = "mean",freeze=0):
        super(CSBERT, self).__init__()

        self.bert = transformers.BertModel.from_pretrained(model_name)
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
"""
