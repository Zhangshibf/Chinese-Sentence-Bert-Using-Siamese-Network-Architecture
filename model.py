import transformers
from torch import nn
from dataloader import create_dataloader


def train_model(epoch,dataloader,model,optimizer):
    loss_f = nn.CosineEmbeddingLoss
    total_loss = 0
    for k in epoch:
        print("-----------------Epoch {}------------------".format(k))

        for batch in dataloader:
            for i in range(len(batch[0])):
                sen_id1 = batch[0][i][0]
                mask1 = batch[1][i][0]
                sen_embeds1 = model(sen_id1, mask1)

                sen_id2 = batch[0][i][1]
                mask2 = batch[1][i][1]
                sen_embeds2 = model(sen_id2, mask2)

                label = batch[2][i]

                loss = loss_f(sen_embeds1,sen_embeds2,label)
                total_loss += loss
                loss.backward()
                optimizer.step()
        avg_loss = total_loss / len(dataloader)
        print(("-----------------Average Loss {}------------------".format(avg_loss)))


def evaluate_model(epoch,dataloader,model,optimizer):
    total_instances = len(dataloader)
    correct_pred = 0
    for k in epoch:
        print("-----------------Epoch {}------------------".format(k))

        for batch in dataloader:
            for i in range(len(batch[0])):
                sen_id1 = batch[0][i][0]
                mask1 = batch[1][i][0]
                sen_embeds1 = model(sen_id1, mask1)

                sen_id2 = batch[0][i][1]
                mask2 = batch[1][i][1]
                sen_embeds2 = model(sen_id2, mask2)

                label = batch[2][i]

                similarity = sen_embeds1*sen_embeds2#I am not sure if this works... I will check it later
                if -1 <= similarity <= -1+2/3:
                    pred = -1
                elif -1+2/3 < similarity < 1-2/3:
                    pred = 0
                elif -1+4/3<= similarity <=1:
                    pred = 1
                else:
                    pred = 100
                if pred == label:
                    correct_pred+=1

        print(("-----------------Accuracy {}------------------".format(correct_pred/total_instances)))



class CSBERT(nn.Module):
    def __int__(self,model_name = "hfl/chinese-bert-wwm",pooling = "mean",out_features = 265,freeze=0):
        super(CSBERT, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(model_name)
        if freeze !=0:
            #freeze bert layers here
            pass
        self.pooling = nn.AvgPool1d(265, stride=265)#need to be changed
        self.linear = nn.Linear(123, out_features = out_features)

    def forward(self,sent_id,mask):
        # pass the inputs to the model
        outputs = self.bert(sent_id, attention_mask=mask)#not sure if the output is correct. Needs to be checked
        pooled = self.pooling(outputs.last_hidden_state)#need to be changed
        sentence_embedding = self.linear(pooled)

        return sentence_embedding


#    def pooling_layer(self,pooling,feature):
#        assert pooling in ["mean","cls"]
#        pooled = list()
#        if pooling == "cls":
#            pooled.append(feature[:, 0])#This could be wrong. Check this later
#        else:
#            pass

#        return pooled
