import argparse
import pickle
import model
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the code')
    parser.add_argument('--test', help="path to pickled test set",required=True)
    parser.add_argument('--device', help="cuda number", required=True)
    parser.add_argument('--model_path', help="path to the model")
    parser.add_argument('--model_name', help="hfl/chinese-macbert-base for MacBert, default bert-wwm",
                        default="hfl/chinese-bert-wwm")
    args = parser.parse_args()

    with open(args.test, 'rb') as pickle_file:
        test_dataloader = pickle.load(pickle_file)


    device_str = "cuda:" + str(args.device)
    device = torch.device(device_str)
    eva_model = model.CSBERT(args.model_name)
    if args.model_path:
        path = args.model_path
        model.load_state_dict(torch.load(path))
    model.evaluate_model_cosine_similarity(test_dataloader, model=eva_model, device=device)