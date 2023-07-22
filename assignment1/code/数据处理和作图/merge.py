import os

# 获取当前脚本所在的文件夹路径
dir_path = os.path.dirname(os.path.abspath(__file__))

# 创建一个新的txt文件，用于保存拼接后的内容
output_file = open(os.path.join(dir_path, 'merged.txt'), 'w')

# 遍历所有子文件夹
for subdir, dirs, files in os.walk(dir_path):
    for file in files:
        # 如果文件名以1000.txt结尾，则将其内容拼接到新的txt文件中
        if file.endswith('.txt'):
            file_path = os.path.join(subdir, file)
            with open(file_path, 'r') as f:
                output_file.write(f.read())

# 关闭文件
output_file.close()