import pandas as pd
from collections import Counter

names = ['v3-zh-wiki-merged-cleaned', 'v3-zh-pp-merged-cleaned', 'v3-zh-norvel-cleaned']
all_text = ''

def save_sta(name, text):
    # 统计每个字出现的频率
    counter = Counter(text)

    # 计算每个字出现的概率
    total_count = sum(counter.values())

    # 将结果转换为DataFrame并按照频率从高至低排序
    df = pd.DataFrame.from_dict(counter, orient='index', columns=['频率'])
    df['概率'] = df['频率'] / total_count
    df = df.sort_values(by='频率', ascending=False)

    # 将结果输出到Excel表格中
    df.to_excel(name + '-freq&prob.xlsx')

for name in names:
    # 读取txt文件
    with open(name + '.txt', 'r') as f:
        text = f.read()
    all_text += text
    save_sta(name, text)

save_sta('zh', all_text)