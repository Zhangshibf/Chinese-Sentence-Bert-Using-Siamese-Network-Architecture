import argparse
import pickle
import model
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the code')
    parser.add_argument('--dev', help="path to dev")
    parser.add_argument('--test', help="path to test")
    parser.add_argument('--device', help="cuda number")
    parser.add_argument('--model_path', help="path to save the result")
    parser.add_argument('--outpath', help="path to save the output")
    parser.add_argument('--model_name', help="hfl/chinese-macbert-base for MacBert, default bert-wwm",
                        default="hfl/chinese-bert-wwm")
    args = parser.parse_args()

    with open(args.dev, 'rb') as pickle_file:
        dev_dataloader = pickle.load(pickle_file)
    with open(args.test, 'rb') as pickle_file:
        test_dataloader = pickle.load(pickle_file)


    device_str = "cuda:" + str(args.device)
    device = torch.device(device_str)
    epoch = 1
    best_model_path = model.evaluate_saved_model(epoch,model_name=args.model_name,model_path=args.model_path
                                           ,dev_dataloader=dev_dataloader ,device=device,outpath = args.outpath)

    print("-----------------Evaluating on Test set------------------")
    best_model = model.CSBERT(model_name=args.model_name)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(device)
    pearson = model.evaluate_model(test_dataloader, best_model, device)
    print("----------------------pearson on Test set {}-----------------------------------".format(pearson))