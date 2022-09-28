import transformers
from torch import nn
import torch
from scipy import stats


def train_model(dataloader,model,optimizer,device,output_path):
    """
    Train CSBERT model for one epoch, then save the trained model.
    :param dataloader: dataloader generated using load_data.py
    :param model: CSBERT model
    :param optimizer: optimizer for training
    :param device: cuda number
    :param output_path: path to save the model
    :return: average loss and average accuracy of this epoch
    """
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

        instance1 = instance[:,0,:].to(device)#first sentence of all sentence pairs
        instance2 = instance[:,1,:].to(device)#second sentences of all sentence pairs
        mask1 = mask[:,0,:].to(device)#masks of first sentence of all sentence pairs
        mask2 = mask[:,1,:].to(device)#masks of second sentences of all sentence pairs

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

    torch.save(model.state_dict(), output_path)
    print("Model saved at {}".format(output_path))

    return avg_loss,avg_accuracy


def evaluate_model_cosine_similarity(dataloader,model,device):
    """
    Evaluate model on the dev set of OCNLI dataset.
    :param dataloader: dataloader of the dev set of OCNLI. It should be generated using load_data.py
    :param model: CSBERT model to be evaluated
    :param device: cuda number
    :return: Spearmans's rho between the cosine-similarities of sentence pairs and the targets
    """
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
            in original targets of dataloader, 0 means entailment, 1 means neutral, and 2 means contradiction
            since we are using consine-similarity for evaluation, we need to convert targets so that -1 means contrafdiction,
            0 means neutral and 1 means entailment
            """
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

    #calculate spearman's rho
    spear = stats.spearmanr(similarity_scores,labels)
    r = spear[0]

    print(("---------Spearman's rank coefficient is {}---------".format(r)))
    return r

def train_and_save_model(epoch,model,optimizer,train_dataloader,device,output_path):
    """
    Train the Chinese Sentence BERT model. Model is saved after each epoch.
    :param epoch: number of epoches
    :param model: CSBERT model
    :param optimizer: optimizer for training
    :param train_dataloader: dataloader generated using load_data.py
    :param device: cuda number
    :param output_path: path to save the model
    :return: loss_list is the list of averaged loss of each epoch, loss_accuracy is the list of averaged accuracy of each epoch
    """
    loss_list = list()
    accuracy_list = list()
    for k in range(epoch):
        print(("-----------------Epoch {}------------------".format(k)))
        print("-----------------Training------------------")
        model.to(device)
        o = str(output_path+str(k)+".pt")
        loss, acc = train_model(train_dataloader, model, optimizer,device,o)
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

def calculate_correct_prediction(outputs,label):
    """
    :param outputs: predictions of CSBERT model
    :param label: targets
    :return: how many predictions are correct
    """
    predictions = torch.argmax(outputs, dim=1).tolist()
    label = label.tolist()
    n = 0
    for i,j in zip(predictions,label):
        if i==j:
            n+=1
    return n
class CSBERT(nn.Module):
    def __init__(self,model_name):
        super(CSBERT, self).__init__()

        self.bert = transformers.BertModel.from_pretrained(model_name)
        self.linear1 = nn.Linear(768, out_features = 300)
        self.linear2 = nn.Linear(900, out_features = 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,sent_id1,mask1,sent_id2,mask2):
        out1 = self.bert(sent_id1, attention_mask=mask1)
        pooled1 = torch.mean((out1[0] * mask1.unsqueeze(-1)), axis=1)#sentences are padded. This step ignores the padded tokens while doing mean pooling
        sentence_embedding1 = self.linear1(pooled1)

        out2 = self.bert(sent_id2, attention_mask=mask2)
        pooled2 = torch.mean((out2[0] * mask2.unsqueeze(-1)), axis=1)
        sentence_embedding2 = self.linear1(pooled2)

        #concatenate u, v and u-v
        embedding_concat = torch.cat((sentence_embedding1, sentence_embedding2, sentence_embedding1 - sentence_embedding2), 1)
        out_linear = self.linear2(embedding_concat)
        prediction = self.softmax(out_linear)

        return prediction

    def generate_sentence_embedding(self,sent_id,mask):
        out = self.bert(sent_id, attention_mask=mask)
        pooled = torch.mean((out[0] * mask.unsqueeze(-1)), axis=1)
        sentence_embedding = self.linear1(pooled)

        return sentence_embedding
