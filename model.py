import pandas
import torch
import transformers
#a
tokenizer = transformers.BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")
model = transformers.BertModel.from_pretrained("hfl/chinese-bert-wwm")
