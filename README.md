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
## Data for training

## Data for testing

## Dataloader

## How to train Chinese Sentence BERT?

## How to test Chinese Sentence BERT?
