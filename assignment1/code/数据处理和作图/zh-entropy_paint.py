import matplotlib.pyplot as plt
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
# calculate the entropy for each corpus
num_points = 19
size = 1
base = size*1024*1024
entropies1 = [calculate_entropy(corpus1[:i*base]) for i in range(1,num_points)]
entropies2 = [calculate_entropy(corpus2[:i*base]) for i in range(1,num_points)]
entropies3 = [calculate_entropy(corpus3[:i*base]) for i in range(1,num_points)]
# create x axis
xarr = list(range(size,num_points*size,size))
# create a figure and axis object
fig, ax = plt.subplots()
# set y axis limitation
#plt.ylim(9, 9.2)
# add last point
last_x = xarr[-1]
for entropy in [entropies1,entropies2,entropies3]:
    last_y = round(entropy[-1],3)
    # 在最后一个点上添加文本注释
    x = last_x-2.5
    y1 = (last_y-0.0005) if (entropy == entropies1) else (last_y+0.0005)
    y2 = (last_y+0.02) if (entropy == entropies2) else (last_y-0.04) if (entropy == entropies3) else (last_y-0.02)
    plt.annotate('{}'.format(last_y), 
                xy=(last_x, y1), 
                xytext=(x, y2), 
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
# add the entropy curves to the plot
ax.plot(xarr, entropies1, label='people')
ax.plot(xarr, entropies2, label='wiki')
ax.plot(xarr, entropies3, label='norvels')
# add a legend and title
ax.legend(loc='lower right')
ax.set_title('Entropy of three Chinese samples')
# x and y labels
plt.xlabel('Number of characters (M)')
plt.ylabel('Entropy')
# display the plot
plt.savefig('zh-entropy' + str(num_points) + '*' + str(size) + '.png', dpi = 1000)