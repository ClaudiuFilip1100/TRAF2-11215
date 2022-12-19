#!/usr/bin/env python
# coding: utf-8

# # Prepare the data
# 
# In order to train everything correctly I need to bring everything to a format understandable by BERT.

# In[1]:


# from datasets import load_dataset

# # dataset = load_dataset("csv", data_dir='', data_files='tokenizer_dataset.csv')
# dataset = load_dataset('conpend_dataset.py')


# In[2]:


# tokens (sequence), ner_tags (sequence)
import pandas as pd

t_df = pd.read_csv('../data/dataset_v2.csv')

combine = lambda x: ' '.join(x['words'])
sentences = t_df.groupby(t_df['sentence #']).apply(combine)
print(sentences[0])

def get_training_corpus():
    for i in range(0, len(sentences)):
        yield sentences[i]


# In[3]:


import numpy as np

dictionar = {'id':[], 'tokens': [], 'tags': []}

def join_sentences(x):
    # print()
    # id = int(np.min(x['id']))
    words = list(x['words'])
    tags = list(x['tag'])

    dictionar['id'].append(int(np.mean(x['sentence #'])))
    dictionar['tokens'].append(words)
    dictionar['tags'].append(tags)


# In[4]:


df = t_df.groupby(['sentence #']).apply(join_sentences)
new_df = pd.DataFrame(dictionar)
new_df.head()


# In[5]:


# shuffle dataset
df_sample = new_df.sample(frac=1)

# split into train / test / validation
df_len = len(df_sample)
train_split = int(0.7 * df_len)
test_split = int(0.15 * df_len)
valid_split = int(0.15 * df_len)

train_df = df_sample.iloc[:train_split,:]
test_df = df_sample.iloc[:test_split,:]
validation_df = df_sample.iloc[:valid_split,:]


# In[6]:


train_df.to_json('data/dataset_bert_train_v2.json', orient='records')
test_df.to_json('data/dataset_bert_test_v2.json', orient='records')
validation_df.to_json('data/dataset_bert_validation_v2.json', orient='records')

