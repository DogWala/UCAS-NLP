import requests
from bs4 import BeautifulSoup
import re

name = 'v3-norvel-zh-zhongwen-unknown'
url_list = []
start_url = []

def wr_txt(file, copy):
    try:
        for subp in copy.find_all('p'):
            text = subp.get_text()
            if text:
                file.write(text)
    except :
        pass

def crawl(current_url, file):
    print(f'正在访问链接：{current_url}')
    # 获取当前链接对应的网页内容
    response = requests.get(current_url)
    html_doc = response.text
    souptxt = BeautifulSoup(html_doc, 'html.parser')
    bodytxt = souptxt.find('div', "content")
    wr_txt(file, bodytxt)
    souphref = BeautifulSoup(html_doc, 'html.parser')
    try:
        link = souphref.find('a', text=re.compile('下一章'))
        href = link.get('href')
        if href:
            crawl(href, file)
    except:
        pass

# 获取网站首页所有书目
url = 'http://www.zongheng.com'
response = requests.get(url)
html_doc = response.text
souptxt = BeautifulSoup(html_doc, 'html.parser')
for link in souptxt.find_all('a'):
    href = link.get('href')
    if href and 'https://book.zongheng.com/book/' in href and 'html' in href:
        url_list.append(href)

# 获取每本书的第一章的链接
for url in url_list:
    response = requests.get(url)
    html_doc = response.text
    souptxt = BeautifulSoup(html_doc, 'html.parser')
    link = souptxt.find('a', {'class':'btn read-btn'})
    href = link.get('href')
    start_url.append(href)

# 打开文件，准备写入文本
with open(name + '.txt', 'w', encoding='utf-8') as file:
    # 爬取入口页面，并将文本写入文件中
    for url in start_url:
        crawl(url, file)