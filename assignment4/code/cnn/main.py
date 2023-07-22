from cnn import TextCnn
import os
import jieba
from torchtext import data
import torch
from tqdm import tqdm

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

# 训练
for i in range(1, epoch + 1):
    loop = tqdm(train_iter)
    for batch in loop:
        text, label = batch.text, batch.label
        text.t_(), label.sub_(1)
        text, label = text.cuda(), label.cuda()
        optimizer.zero_grad()
        output = cnn(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        loop.set_description('Epoch [{}/{}]'.format(i, epoch))
        step += 1
        steps.append(step)
        corrects = (torch.max(output, 1)[1].view(
                label.size()).data == label.data).sum()
        accuracy = 100.0 * float(corrects) / batch.batch_size
        loss_train.append(loss.item())
        acc_train.append(accuracy)
    test_acc, test_loss = test(test_iter, cnn)
    acc_test.append(test_acc)
    if test_acc > acc_best:
        acc_best = test_acc
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(cnn.state_dict(), path + '/model_best.pt')
    test_steps.append(step)


# 将训练过程记录绘制成图像
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5), dpi = 200)
plt.plot(steps, acc_train, label = 'train')
plt.plot(test_steps, acc_test, label = 'test')
plt.xlabel('steps')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('accuracy.png', dpi = 200)

plt.figure(figsize=(10, 5), dpi = 200)
plt.plot(steps, loss_train, label = 'train')
plt.xlabel('steps')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss.png', dpi = 200)

# 测试
test(test_iter, cnn)