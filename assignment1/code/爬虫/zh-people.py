import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

name = 'v3-zh-pp-10*1000'
website = 'http://www.people.com.cn'
max = 10000
visited = set() # 存储已经访问过的链接

def wr_txt(file, copy):
    for subp in copy.find_all('p'):
        text = subp.get_text()
        if text:
            file.write(text)

def bfs(url, file):
    """
    广度优先遍历所有链接，并获取对应网页的内容
    """
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
                html_doc = response.content.decode('gbk')
                souptxt = BeautifulSoup(html_doc, 'html.parser')
                bodytext = souptxt.find('div', "rm_txt_con cf")
                if bodytext:
                    print(counter)
                    wr_txt(file,bodytext)
                    counter = counter + 1
                if len(queue) < max:
                    # 获取当前链接中的所有链接
                    souphref = BeautifulSoup(html_doc, 'html.parser')
                    for link in souphref.find_all('a'):
                        href = link.get('href')
                        if href and 'people.com.cn' in href and 'html' in href:
                            href = urljoin(current_url, href)
                            queue.append(href)
            except:
                pass


# 打开文件，准备写入文本
with open(name + '.txt', 'w', encoding='utf-8') as file:
    # 爬取入口页面，并将文本写入文件中
    bfs(website, file)