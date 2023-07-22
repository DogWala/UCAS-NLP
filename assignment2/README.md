# 自然语言处理大作业2——词向量

##### 随机二十个词最相近的十个词

保存在每个以模型命名的文件夹中，名称为

`model`-`epo{EPOCH}ebd{EMBEDDING_SIZE}vcb{VOCAB_SIZE}.xlsx` 或

`model`-`epo{EPOCH}ebd{EMBEDDING_SIZE}vcb{VOCAB_SIZE}win{WINDOW_SIZE}.xlsx` 



##### 语料

训练使用的语料置于每个类型语料的根最外层文件夹中。训练时需要放入模型对应的文件夹中才能使用。



##### 训练

训练使用的参数在文件定义。



##### 模型使用

`{model_name}_use.ipynb` 可以查询与输入词的词向量最接近的 10 个词，需要与训练使用的参数统一。



##### 结果保存

使用 `saveRes.ipynb` 保存从语料中抽取的 20 个词最接近的 10 个词以及他们之间的相似度。