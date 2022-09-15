import argparse
import pickle
import model
import torch
from torch import optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--train',help = "path to train")
    parser.add_argument('--device',help = "cuda number")
    parser.add_argument('--model_name', help="hfl/chinese-macbert-base for MacBert, default bert-wwm",default="hfl/chinese-bert-wwm")
    args = parser.parse_args()

    with open(args.train, 'rb') as pickle_file:
        train_dataloader = pickle.load(pickle_file)

    epoch = 200
    csbert_model = model.CSBERT(model_name=args.model_name)
    optimizer = optim.SGD(csbert_model.parameters(), lr=0.001, momentum=0.9)
    device_str = "cuda:"+str(args.device)
    device = torch.device(device_str)

    model.train_and_save_model(epoch,csbert_model,optimizer,train_dataloader,device)
