import pandas as pd
# 将ChnSentiCorp_htl_all.csv分为train.tsv和test.tsv
df = pd.read_csv('ChnSentiCorp_htl_all.csv')
df.dropna(inplace=True)
labels = df['label']
texts = df['review']
from sklearn.model_selection import train_test_split
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.1)
train = pd.DataFrame({'label': train_labels, 'text': train_texts})
test = pd.DataFrame({'label': test_labels, 'text': test_texts})
train.to_csv('train.tsv', sep='\t', index=False)
test.to_csv('test.tsv', sep='\t', index=False)
