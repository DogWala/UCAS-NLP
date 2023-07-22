import math
def calculate_entropy(data):
    entropy = 0
    total_chars = len(data)
    freq_dict = {}
    for char in data:
        if char in freq_dict:
            freq_dict[char] += 1
        else:
            freq_dict[char] = 1
    for freq in freq_dict.values():
        prob = freq / total_chars
        entropy += prob * math.log2(prob)
    return -entropy
# read the text from each corpus
corpus1 = open('v3-zh-pp-merged-cleaned.txt').read()
corpus2 = open('v3-zh-wiki-merged-cleaned.txt').read()
corpus3 = open('v3-zh-norvel-cleaned.txt').read()
# caculate entropy
corpus = corpus1 + corpus2 + corpus3
entropies = calculate_entropy(corpus)
print(entropies)