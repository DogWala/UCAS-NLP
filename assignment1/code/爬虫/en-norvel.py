import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin

name = 'v3-en-norvel-fulleng'
url_list = []
start_url = []

def wr_txt(file, copy):
    #try:
        for subp in copy.find_all('p'):
            text = subp.get_text()
            if text:
                file.write(text)
    #except :
    #    pass

def crawl(current_url, file):
    try:
        print(f'正在访问链接：{current_url}')
        # 获取当前链接对应的网页内容
        response = requests.get(current_url)
        html_doc = response.text
        souptxt = BeautifulSoup(html_doc, 'html.parser')
        bodytxt = souptxt.find('div', "col-xs-12 inner-content")
        wr_txt(file, bodytxt)
        souphref = BeautifulSoup(html_doc, 'html.parser')
        link = souphref.find('a', text='»')
        href = urljoin(current_url, link.get('href'))
        if href:
            crawl(href, file)
    except:
        pass

# 获取网站首页所有书目
baseurl = 'https://full-english-books.net/english-books/'
for i in range(17, 50):
    try:
        url = baseurl + str(i)
        print(url)
        response = requests.get(url)
        html_doc = response.text
        souptxt = BeautifulSoup(html_doc, 'html.parser')
        for link in souptxt.find_all('a'):
            href = link.get('href')
            if href and '/english-books/' in href and 'read-online' in href:
                href = urljoin(url,href)
                url_list.append(href)
    except:
        pass

# 打开文件，准备写入文本
with open(name + '.txt', 'a', encoding='utf-8') as file:
    # 爬取入口页面，并将文本写入文件中
    for url in url_list:
        crawl(url, file)