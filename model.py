import pandas
import torch
import transformers
import json
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from sentence_transformers import SentenceTransformer, models, losses



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