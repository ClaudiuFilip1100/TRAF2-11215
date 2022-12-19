#!/usr/bin/env python
# coding: utf-8

# # Importing
# 
# Pytorch data: Dataset and Dataloader.
# - create a CustomDataset class: contains encased preprocessing
# - Dataloader: feeds the data in batches
# - the role of Dataset and Dataloader classes is to encapsulate the preprocessing steps
# ## CustomDataset
# - accepts tokenizer, dataframe and max_length as input -> generates tokenized output and tags used by BERT.
# 
# ## Dataloader
# - defines how to data is loaded into the NN (batching)
# 
# 
# Q: Should I use BertTokenizer? Or use my own tokenizer? `BertTokenizer.from_pretrained('bert-base-uncased')`
# A: No, I will build my own tokenizer.
# 
# Q: If I DO use the BertTokenizer what will the id's for the new things be? In our vocabulary we have quite a few non-words, so will the BertTokenizer act?
# A: When building the tokenizer, I'll build to vocabulary as well.

# In[14]:


# source https://huggingface.co/course/chapter6/8?fw=pt

import pandas as pd

t_df = pd.read_csv('../data/dataset_v2.csv')

# t_df = t_df.replace({'<QUANTITY>':'[QUANTITY]',
#                     '<UNITPRICEAMOUNT>':'[UNITPRICEAMOUNT]',
#                     '<DATE>':'[DATE]',
#                     '<INCOTERMS>':'[INCOTERMS]'})

combine = lambda x: ' '.join(x['words'])
sentences = t_df.groupby(t_df['sentence #']).apply(combine)
print(sentences[0])

def get_training_corpus():
    for i in range(0, len(sentences)):
        yield sentences[i]


# In[15]:


sentences.head()


# In[21]:


# we start by instantiating a Tokenizer object with a model, 
# then set its normalizer, 
# pre_tokenizer, 
# post_processor, 
# and decoder attributes to the values we want.
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

example = 'CPT LAHORE AIRPORT PAKISTAN QTY [QUANTITY] PC OF MULTI MODE READER/TRINOCULAR MICROSCOPE SYSTEM MODEL'

tokenizer = Tokenizer(models.WordPiece(unk_token='[UNK]'))
tokenizer.normalizer = normalizers.BertNormalizer()
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[UNK]']
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)


# In[22]:


# train the tokenizer

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
cls_token_id = tokenizer.token_to_id('[CLS]')
sep_token_id = tokenizer.token_to_id('[SEP]')
unk_token_id = tokenizer.token_to_id('[UNK]')
cls_token_id, sep_token_id, unk_token_id


# In[23]:


# define the post-processing BERT template
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id),("[SEP]", sep_token_id)]
)
tokenizer.decoder = decoders.WordPiece()


# In[24]:


encodings = tokenizer.encode(example)
encodings.tokens


# In[25]:


# we need to define the BERT template
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id),("[SEP]", sep_token_id)]
)


# In[26]:


tokenizer.decoder = decoders.WordPiece()
tokenizer.decode(encodings.ids)


# In[27]:


tokenizer.save('tokenizer/tokenizer_v2.json')

