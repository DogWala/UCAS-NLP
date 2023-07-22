import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import BertTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from mydataset import MyDataset

epoch = 20
batch_size = 8
lr = 0.000007
name = 'models3'

train_pd = pd.read_csv('train.tsv', sep='\t')
test_pd = pd.read_csv('test.tsv', sep='\t')

train_texts = train_pd['text']
train_labels = train_pd['label']
test_texts = test_pd['text']
test_labels = test_pd['label']

# 初始化tokenizer和model
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese')

# 保存模型的文件夹
if not os.path.exists(name):
    os.mkdir(name)

# 重置索引
train_texts, test_texts = train_texts.reset_index(drop=True), test_texts.reset_index(drop=True)
train_labels, test_labels = train_labels.reset_index(drop=True), test_labels.reset_index(drop=True)

# 创建训练数据集和数据加载器
train_data = MyDataset(train_texts, train_labels, tokenizer, max_len=128)
train_dataloader = DataLoader(train_data, batch_size=batch_size)

# 创建测试数据集和数据加载器
test_data = MyDataset(test_texts, test_labels, tokenizer, max_len=128)
test_dataloader = DataLoader(test_data, batch_size=1)

# 训练模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

steps = []
loss_train = []

acc_test = []
step_test = []

acc_best = 0
step = 0
# 在原有训练循环的基础上添加 tqdm() 包装 train_dataloader
for epoch in range(epoch):
    loop = tqdm(train_dataloader, total=len(train_dataloader), leave=True)  # 添加这行
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        # newly added
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新进度条
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        steps.append(step)
        step += 1
        loss_train.append(loss.item())
    # 在每个epoch结束后，测试模型
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_count = 0
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # 获取预测结果，这是在logits上取argmax
            predictions = torch.argmax(outputs.logits, dim=-1)

            # 计算正确预测的数量
            total_correct += (predictions == labels).sum().item()
            total_count += labels.size(0)

        # 计算准确率
        accuracy = total_correct / total_count
        acc_test.append(accuracy)
        step_test.append(step)
        # 打印准确率
        print(f'Accuracy: {accuracy * 100:.2f}%')
        # 保存检查点
        if accuracy > acc_best:
            acc_best = accuracy
            torch.save(model.state_dict(), f'{name}/best-acc-{accuracy * 100:.2f}.pth')
        torch.save(model.state_dict(), f'{name}/checkpoint-{epoch}-acc-{accuracy * 100:.2f}.pth')

# 绘制训练过程中的loss曲线
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5), dpi = 200)
plt.plot(steps, loss_train)
plt.xlabel('step')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss20.png')

# 绘制测试过程中的accuracy曲线
plt.figure(figsize=(10, 5), dpi = 200)
plt.plot(step_test, acc_test)
plt.xlabel('step')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('accuracy20.png')