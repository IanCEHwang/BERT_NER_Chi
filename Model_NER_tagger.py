#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import transformers
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from transformers import AutoModelForTokenClassification
from transformers import pipeline
from transformers import BertTokenizer



data = np.load('./Training_data_cleaned/cleaned_data_np.npy',allow_pickle=True)
token = data[0]
token_list = [sentence.tolist() for sentence in token]

### read data from pre-tagged data
unique_tags = set(tag for doc in data[1] for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}
label_list=list(unique_tags)


# ### Import Model and Tokenizer
trained_model = 'bert-base-chinese-30epoch-64batch_ES' #wanted model name

model = AutoModelForTokenClassification.from_pretrained(f"./model/{trained_model}", num_labels=len(label_list))
my_tokenizer = BertTokenizer.from_pretrained(f'./model/{trained_model}//')


# ### Name Entity Recognition Model import
ner = pipeline("ner", model=model, tokenizer= my_tokenizer,grouped_entities = True,\
               aggregation_strategy = "simple" )
#ner pipeline setting 
#1.group_entities = whether to group consecutive words with same predicted labels
#2.aggregation_strategy = Will attempt to group entities following the default schema.\
# (A, B-TAG), (B, I-TAG), (C, I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being [{“word”: ABC, “entity”: “TAG”}


# ## Test model
def print_result(sample):
    tagger_sample = ner(sample)
    for tags in tagger_sample:
        print(f"{tags['entity_group']} : {tags['word']}")


# # #### test 1
# sample_sentence = '光泉優酪乳1250ml'
# print_result(sample_sentence)


# # #### test 2
# sample_sentence = '義美手工餅乾3包入'
# print_result(sample_sentence)


# # test 3
# sample_sentence = '鬼金棒拉麵一碗'
# print_result(sample_sentence)

print("Please type in sample receipt name here:")
sample = input()
print_result(sample)
