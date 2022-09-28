# Chinese-Sentence-Bert-Using-Siamese-Network-Architecture

Please refer to [this report](https://drive.google.com/file/d/1VIoPOra21WvKlv8C4Ehp7kX2JxkGbwwI/view?usp=sharing) for details.

-----------------------------------------------------------
# How to use Chinese Sentence BERT to generate sentence embeddings?

Download Chinese Sentence BERT model from [here](https://drive.google.com/file/d/1ctyI2eRZVDXKRSCuiEsmFVT5Onrr81ZX/view?usp=sharing).

```
import model
import transformer

#tokenize the sentences
tokenizer = transformers.BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")
tokens = tokenizer.batch_encode_plus(your_sentence, padding=True, return_token_type_ids=False)
sentence = tokens['input_ids']
mask = tokens['attention_mask']
    
#load sentence bert
sentence_bert_model = model.CSBERT()
sentence_bert_model.load_state_dict(torch.load(path_to_the_Chinese_Sentence_BERT_model))

#generate sentence embeddings
sentence_embeddings = sentence_bert_model.generate_sentence_embedding(sentence,mask)
```

------------------------------------------------------------
# Alternatively, if you'd like to try training Chinese Sentence BERT yourself...

## Set up environment

. .bashrc

conda env create -f env_csbert.yml

conda activate csbert

## Data for training
Please download train set of Original Chinese Natural Language Inference dataset from [here](https://github.com/CLUEbenchmark/OCNLI/tree/main/data/ocnli)

Please download Chinese Multi-Genre NLI dataset from [here](https://storage.googleapis.com/cluebenchmark/tasks/cmnli_public.zip)
## Data for testing
Please download dev set of Original Chinese Natural Language Inference dataset from [here](https://github.com/CLUEbenchmark/OCNLI/tree/main/data/ocnli)
## Dataloader
```
python dataloader.py --i path_to_the_dataset --o path_to_save_the_dataloader
```
## How to train Chinese Sentence BERT?
```
python train.py --train path_to_the_dataloader_of_train_set --out path_to_save_trained_model --device the_cuda_number_you'd_like_to_use
```
## How to test Chinese Sentence BERT?
```
python evaluate.py --test path_to_the_dataloader_of_test_set --device the_cuda_number_you'd_like_to_use --model_path path_to_save_trained_model
```
