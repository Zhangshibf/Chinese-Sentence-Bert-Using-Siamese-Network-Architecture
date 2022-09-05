import argparse
import pickle
import model
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the code')
    parser.add_argument('--dev', help="path to dev")
    parser.add_argument('--test', help="path to test")
    parser.add_argument('--device', help="cuda number")
    args = parser.parse_args()

    with open(args.dev, 'rb') as pickle_file:
        dev_dataloader = pickle.load(pickle_file)
    with open(args.test, 'rb') as pickle_file:
        test_dataloader = pickle.load(pickle_file)


    device_str = "cuda:" + str(args.device)
    device = torch.device(device_str)

    epoch = 1#remember to change here
    best_model_path = model.evaluate_saved_model(epoch,model_path="/home/CE/zhangshi/mygithubprojects/csbert/"
                                           ,dev_dataloader=dev_dataloader ,device=device)

    print("-----------------Evaluating on Test set------------------")
    best_model = model.CSBERT()
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(device)
    loss, acc = model.evaluate_model(test_dataloader, best_model, device)

    print("----------------------Accuracy on Test set {}-----------------------------------".format(acc))