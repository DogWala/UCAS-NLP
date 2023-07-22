import requests
from bs4 import BeautifulSoup

name = 'v3-abc'
max = 4000
website = 'https://abcnews.go.com'

def wr_txt(file, copy):
    try:
        for dlt in copy.find_all('span'):
            dlt.decompose()
        for link in copy.find_all('a'):
            link.replace_with(link.text)
        for subp in copy.find_all('p'):
            subp.extract()
            text = subp.text.strip()  # 获取标签中的文本，去除两端的空格
            if text:
                file.write(text)
    except :
        pass

def bfs(url, file):
    """
    广度优先遍历所有链接，并获取对应网页的内容
    """
    visited = set() # 存储已经访问过的链接
    queue = [] # 存储待访问的链接
    queue.append(url)

    counter = 0  # 记录已经爬取的次数

    while queue:
        if counter >= max:
            break
        current_url = queue.pop(0)
        if current_url not in visited:
            try:
                visited.add(current_url)
                print(f'正在访问链接：{current_url}')
                # 获取当前链接对应的网页内容
                response = requests.get(current_url)
                html_doc = response.text
                souptxt = BeautifulSoup(html_doc, 'html.parser')
                bodytxt = souptxt.find('div', {'class': lambda x: x and 'main__body' in x})
                print(counter)
                wr_txt(file, bodytxt)
                counter = counter + 1
                # 获取当前链接中的所有链接
                llink = []
                souphref = BeautifulSoup(html_doc, 'html.parser')
                for link in souphref.find_all('a'):
                    href = link.get('href')
                    if href and 'abcnews.go.com' in href:
                        llink.append(href)
                for link in llink:
                    if link not in visited:
                        queue.append(link)
            except:
                pass

# 打开文件，准备写入文本
with open(name + '.txt', 'w', encoding='utf-8') as file:
    # 爬取入口页面，并将文本写入文件中
    bfs(website, file)