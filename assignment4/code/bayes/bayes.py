import pandas as pd
import jieba
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# 从tsv文件中读取数据
def read_tsv(path):
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = ['label', 'text']
    return df

# 读取训练集和测试集
train = read_tsv('train.tsv')
test = read_tsv('test.tsv')

# jieba分词
def tokenizer(text):
    return [word for word in jieba.cut(text) if word.strip()]

# 停用词，从文件中读取
stop_words = []
with open('stop.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        stop_words.append(line.strip())

# 计算tf-idf
tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words=stop_words)
train_features = tfidf.fit_transform(train['text'])
test_features = tfidf.transform(test['text'])

# 保存关键词，以及对应的权重
with open('keywords.txt', 'w', encoding='utf-8') as f:
    for k, v in tfidf.vocabulary_.items():
        f.write(k + ' ' + str(tfidf.idf_[v] * 10000) + '\n')

# 按照权重排序，重新保存
with open('keywords.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines.sort(key=lambda x: float(x.split(' ')[1]), reverse=True)
    with open('keywords.txt', 'w', encoding='utf-8') as f2:
        f2.writelines(lines)

# 这个权重意味着什么？权重越大，越重要，越能区分不同类别的文本

# labels 
train_labels = train['label']
test_labels = test['label']

# 训练模型
clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)

# 预测
predicted_labels = clf.predict(test_features)

# 评估
print('准确率为：', accuracy_score(test_labels, predicted_labels))

# 将错误分样本的标签和文本保存到文件中，保存原标签，预测标签，文本
test['predicted'] = predicted_labels
with open('error.txt', 'w', encoding='utf-8') as f:
    for label, predicted, text in zip(test['label'], test['predicted'], test['text']):
        if label != predicted:
            f.write(str(label) + '\t' + str(predicted) + '\t' + text + '\n')