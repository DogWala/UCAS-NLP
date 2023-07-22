import jieba
# 加载停用词表
stopwords = ['的', '地', '得', '了', '着', '就', '也', '都', '和', '与', '在', '向', '为', '是', '有', '无', '很', '多', '些', '此', '这', '那', '或', '即', '等', '诸', '或者', '虽', '然', '但', '并', '而', '且', '究竟', '是否', '因为', '所以', '以及', '而且', '从而', '表示', '并且', '同时', '只有', '按照', '而言', '可以', '关于', '据此', '唯独', '总的来看', '在此基础上', '如果', '例如', '尤其是', '甚至', '为了', '以至于', '根据', '并非', '常常', '原来', '之所以', '即使', '以免', '不论', '即便', '反之', '否则', '随着', '与此同时', '然而', '可是', '换句话说', '事实上', '比方说', '譬如', '毕竟', '就是说', '也就是说', '概括说', '如下', '举例来说', '不过', '然后', '不光', '至于', '基本上', '虽然', '不少', '故', '另外', '别', '除了', '兼', '偏偏', '偷偷', '别的', '各', '另', '两者', '同']
# 加载样本
names = ['v3-zh-wiki-merged-cleaned', 'v3-zh-pp-merged-cleaned', 'v3-zh-norvel-cleaned']
all_text = ''
for name in names:
    # 读取txt文件
    with open(name + '.txt', 'r') as f:
        text = f.read()
    all_text += text
# 分词并统计词频
word_count = {}
for word in jieba.cut(all_text):
    if word not in stopwords:
        word_count[word] = word_count.get(word, 0) + 1
# 按照词频排序
sorted_word_count = list(sorted(word_count.items(), key=lambda x: x[1], reverse=True))
# 打印前10个实词
for word, count in sorted_word_count[:10]:
    print(word, count)
print(len(words_list))