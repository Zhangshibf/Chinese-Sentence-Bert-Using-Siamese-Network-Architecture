import argparse
import pickle
import model
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the code')
    parser.add_argument('--dev', help="path to dev")
    parser.add_argument('--test', help="path to test")
    parser.add_argument('--device', help="cuda number")
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
    model.evaluate_model_cosine_similarity(dev_dataloader, model=args.model_name, device=args.device)
    model.evaluate_model_cosine_similarity(test_dataloader, model=args.model_name, device=args.device)