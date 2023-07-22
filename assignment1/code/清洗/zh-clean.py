# 繁体转简体
import opencc
import re

name = 'wiki-zh-merged'
# 创建OpenCC对象，指定转换规则
converter = opencc.OpenCC('t2s')

# 打开输入文件和输出文件
input_file = open(name + '.txt', 'r', encoding='utf-8')
output_file = open(name + '-cleaned.txt', 'w', encoding='utf-8')

# 删除任何非汉字符号
# 定义正则表达式，匹配除了中文以外的任何字符
pattern = re.compile(r'[^\u4e00-\u9fa5]')

# 逐行读取输入文件内容，并将每行繁体字转换为简体字，然后写入输出文件
for line in input_file:
    simplified_line = converter.convert(line.strip())
    cleaned_line = re.sub(pattern, '', simplified_line.strip())
    output_file.write(cleaned_line)

# 关闭文件
input_file.close()
output_file.close()