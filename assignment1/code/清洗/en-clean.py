# 删除英文中的符号，不保留空格
import re

# 打开输入文件和输出文件
name = 'v3-en-wiki-merged'
input_file = open(name + '.txt', 'r', encoding='utf-8')
output_file = open(name + '-cleaned.txt', 'w', encoding='utf-8')

# 读取文本文件
text = input_file.read()

text = re.sub(r'[^a-zA-Z\s]', '', text)
text = re.sub(r"\s+", " ", text).strip()

# 大写转小写
text = text.lower()

output_file.write(text)

# 关闭文件
input_file.close()
output_file.close()