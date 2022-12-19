#!/usr/bin/env python
# coding: utf-8

# # Import dataset

# In[1]:


# source https://huggingface.co/course/chapter7/2?fw=tf
import datasets
from datasets import load_dataset

classes = ["O", "Quantity", "UnitPriceAmount", "GoodsDescription",
            "Incoterms", "GoodsOrigin", "Tolerance", "HSCode"]

dataset = load_dataset("json", data_files={'train':'data/dataset_bert_train_v2.json', 'test':'data/dataset_bert_test_v2.json', 'validation':'data/dataset_bert_validation_v2.json'}, features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "tags": datasets.Sequence(datasets.features.ClassLabel(names=classes))
        }))
dataset


# ## Example

# In[2]:


words = dataset["train"][0]["tokens"]
labels = dataset["train"][0]["tags"]
line1 = ""
line2 = ""
for word, label in zip(words, labels):
    full_label = classes[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)


# In[3]:


print(dataset["train"][0]["tokens"])


# # Load Tokenizer

# In[4]:


# LOAD TOKENIZER
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer/tokenizer_v2.json",
    bos_token="[S]",
    eos_token="[/S]",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
    padding_side="right",
    model_max_len=300
)

inputs = tokenizer(dataset["train"][0]["tokens"], is_split_into_words=True)
print(inputs.tokens())


# In[5]:


print(inputs.word_ids())


# # Aling Tokens with Values

# In[6]:


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]

    return new_labels


# In[7]:


labels = dataset["train"][0]["tags"]
word_ids = inputs.word_ids()
print(labels)
print(align_labels_with_tokens(labels, word_ids))


# In[8]:


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True,
        max_length=300,
    )
    all_labels = examples["tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


# In[9]:


tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
)


# ## Add Padding

# In[10]:


from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer, return_tensors="tf", padding='max_length', max_length=300
)


# In[11]:


batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
batch["labels"]


# In[12]:


features = tokenized_datasets['train'].features
# label_name = "label" if "label" in features[0].keys() else "labels"
# labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
features.keys()
# tokenizer.pad(tokenized_datasets['train'].features, padding=tokenizer.padding_side)


# In[13]:


for i in range(2):
    print(tokenized_datasets["train"][i]["labels"])


# In[14]:


tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=16,
)

tf_eval_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)


# In[15]:


id2label = {i: label for i, label in enumerate(classes)}
label2id = {v: k for k, v in id2label.items()}


# # Train

# In[16]:


from transformers import TFAutoModelForTokenClassification

model = TFAutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    id2label=id2label,
    label2id=label2id,
    max_length=300
)


# In[17]:


model.summary()


# In[18]:


from transformers import create_optimizer
import tensorflow as tf

# Train in mixed-precision float16
# Comment this line out if you're using a GPU that will not benefit from this
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
# by the total number of epochs. Note that the tf_train_dataset here is a batched tf.data.Dataset,
# not the original Hugging Face Dataset, so its len() is already num_samples // batch_size.
num_epochs = 10
num_train_steps = len(tf_train_dataset) * num_epochs

optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)


# In[19]:


from huggingface_hub import notebook_login

notebook_login()
# token: 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


# In[20]:


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[21]:


from transformers.keras_callbacks import PushToHubCallback

callback = PushToHubCallback(output_dir="bert-ner-conpend-v4", tokenizer=tokenizer)

history = model.fit(
    tf_train_dataset,
    validation_data=tf_eval_dataset,
    callbacks=[callback],
    epochs=num_epochs,
)


# In[22]:


import evaluate

metric = evaluate.load("seqeval")


# In[23]:


import numpy as np

all_predictions = []
all_labels = []
for batch in tf_eval_dataset:
    logits = model.predict_on_batch(batch)["logits"]
    labels = batch["labels"]
    predictions = np.argmax(logits, axis=-1)
    for prediction, label in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:
                continue
            all_predictions.append(classes[predicted_idx])
            all_labels.append(classes[label_idx])
metric.compute(predictions=[all_predictions], references=[all_labels])


# In[24]:


model.push_to_hub('bert-ner-conpend-v4')
tokenizer.push_to_hub('bert-ner-conpend-v4')

