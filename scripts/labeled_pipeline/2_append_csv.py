import pandas as pd

df_1 = pd.read_csv('../../data/dataset_from_json.csv')
df_2 = pd.read_csv('../../data/dataset_from_json_v2.csv')

# change the sentence number of df_2
n_sentences = len(df_1.groupby(df_1['sentence #']).count())
n_sentences_append = len(df_2.groupby(df_2['sentence #']).count())

#old_indices
old_indices = list(range(n_sentences_append))
# new_sentences
new_indices = list(range(n_sentences, n_sentences+n_sentences_append))

df_2.replace(old_indices, new_indices, inplace=True)

df = df_1.append(df_2)
df.to_csv('../../data/dataset_v2.csv')