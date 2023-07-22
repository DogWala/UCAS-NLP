import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import BertTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader
from mydataset import MyDataset

epoch = 10
batch_size = 16
lr = 0.00001

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
name = 'models1'

# 重置索引
train_texts, test_texts = train_texts.reset_index(drop=True), test_texts.reset_index(drop=True)
train_labels, test_labels = train_labels.reset_index(drop=True), test_labels.reset_index(drop=True)

# 创建测试数据集和数据加载器
test_data = MyDataset(test_texts, test_labels, tokenizer, max_len=128)
test_dataloader = DataLoader(test_data, batch_size=1)

# 训练模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 加载最优模型，在测试集上测试，将预测错误的文本保存到error.txt
model.load_state_dict(torch.load(f'{name}/best-acc-90.73.pth'))
model.to(device)
model.eval()
with torch.no_grad():
    with open('error.txt', 'w', encoding='utf-8') as f:
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # 获取预测结果，这是在logits上取argmax
            predictions = torch.argmax(outputs.logits, dim=-1)

            # 将预测错误的文本保存到error.txt
            for text, label, prediction in zip(batch['text'], labels, predictions):
                if label != prediction:
                    # convert to int and then to str
                    label = str(int(label))
                    prediction = str(int(prediction))
                    f.write(label + '\t' + prediction + '\t' + text + '\n')