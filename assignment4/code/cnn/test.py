from cnn import TextCnn
import os
import jieba
from torchtext import data
import torch
from tqdm import tqdm
import pandas as pd

# parameters
epoch = 20 
batch_size = 64
lr = 0.0007
path = "models"

# jieba分词
def tokenizer(text):
    return [word for word in jieba.cut(text) if word.strip()]

# 读取数据
text_field = data.Field(lower=True, tokenize = tokenizer)
label_field = data.Field(sequential=False)
filds = [('label', label_field), ('text', text_field)]
train_dataset, test_dataset = data.TabularDataset.splits(
    path = '', format = 'tsv', skip_header = False,
    train = 'train.tsv', test = 'test.tsv', fields = filds
)

# 构建词典
text_field.build_vocab(train_dataset, test_dataset, min_freq = 5, max_size = 50000)
label_field.build_vocab(train_dataset, test_dataset)
# 构建迭代器
train_iter, test_iter = data.Iterator.splits((train_dataset, test_dataset),
                            batch_sizes = (batch_size, batch_size), sort_key = lambda x: len(x.text))

# 模型参数
embed_num = len(text_field.vocab)
class_num = len(label_field.vocab) - 1
kernel_sizes = [3, 4, 5]
embed_dim = 128
kernel_num = 10
dropout = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型
cnn = TextCnn(embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout)
cnn.cuda()

# 优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

# 损失函数
criterion = torch.nn.CrossEntropyLoss()

# 记录
steps = []
test_steps = []
acc_train = []
acc_test = []
acc_best = 0
loss_train = []
step = 0
interval = 20

# 测试
def test(data_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)  # batch first, index align
        feature, target = feature.cuda(), target.cuda()
        
        logit = model(feature)
        loss = criterion(logit, target)

        avg_loss += loss.data
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * float(corrects) / size
    return float(accuracy), float(avg_loss)

# 读取最佳模型
cnn.load_state_dict(torch.load(os.path.join(path, 'model_best.pt')))
# 测试
# acc, loss = test(test_iter, cnn)
# print('test accuracy:{:.2f}%'.format(acc))

# 将预测错误的原始文本保存到error.txt，直接从csv文件中读取文本
test = pd.read_csv('test.tsv', sep='\t')
test_text = test['text']
test_label = test['label']

with open('error.txt', 'w', encoding='utf-8') as f:
    for text, label in zip(test_text, test_label):
        feature = text_field.preprocess(text)
        feature = [[text_field.vocab.stoi[x] for x in feature]]
        feature = torch.LongTensor(feature)
        feature = feature.cuda()
        logit = cnn(feature)
        pred = torch.max(logit, 1)[1].view(1).data
        # convert to int
        label = int(label)
        pred = int(pred)
        if pred != label:
            f.write(str(label) + '\t' + str(pred) + '\t' + text + '\n')