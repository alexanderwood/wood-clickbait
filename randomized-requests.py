from lxml.html import fromstring
import requests
from itertools import cycle
import traceback
from retrying import retry
from bs4 import BeautifulSoup

def extract_metadata_nyt(dat, tweet_id):
    '''Extract and save metadata'''
    soup = BeautifulSoup(dat.text, parser='lxml', features='lxml')
    values = [tweet_id, dat.url]
    values.append(soup.find_all("meta",attrs={"name":"articleid"})[0].attrs['content'])
    values.append(soup.find_all("meta",attrs={"property":"article:published"})[0].attrs['content'])
    values.append(soup.find_all("meta",attrs={"property":"twitter:url"})[0].attrs['content'])
    values.append(soup.find_all("meta",attrs={"property":"og:title"})[0].attrs['content'])
    values.append(soup.find_all("meta",attrs={"property":"twitter:title"})[0].attrs['content'])
    values.append(soup.find_all("meta",attrs={"property":"article:tag"})[0].attrs['content'])
    values.append(soup.find_all("meta",attrs={"name":"news_keywords"})[0].attrs['content'])
    return values


def get_proxies():
    url = 'https://free-proxy-list.net/'
    response = requests.get(url)
    parser = fromstring(response.text)
    proxies = set()
    for i in parser.xpath('//tbody/tr')[:10000]:
        if i.xpath('.//td[7][contains(text(),"yes")]'):
            proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
            proxies.add(proxy)
    return proxies


@retry(stop_max_attempt_number=2, wait_fixed=1000)
def query_site(proxy, url):
    try:
        response = requests.get(url, proxies=proxy)
    except:
        response = None
    return response


def get_proxy_info():
    proxy_info = collector.get_proxy({'anonymous':True})
    proxy = {
        'http': '{}:{}'.format(proxy_info.host, proxy_info.port),
        'https': '{}:{}'.format(proxy_info.host, proxy_info.port)
    }

    return proxy_info, proxy


def save_info(payload, row, data_dir, handle):
    tweet_id = row[0]
    if payload:
        values = extract_metadata_nyt(payload, tweet_id)
        with open(data_dir / "web-data-{}.csv".format(handle), "a") as fout:
            writer = csv.writer(fout)
            writer.writerow(values)
    else:
        print("\nFailed.")
        with open(data_dir / "web-data-err.txt", 'a') as fout:
            writer = csv.writer(fout)
            writer.writerow(tweet_id)


def try_requesting(row):
    url = row[-1]

    for i in range(10):
        print(i)

        proxy_info, proxy = get_proxy_info()

        payload = query_site(proxy, url)
        if payload:
            print("\nSuccess:", payload)
            break
        else:
            collector.remove_proxy(proxy_info)

    return payload


# Retrieve any http proxy
#proxy = collector.get_proxy()

# Retrieve only 'us' proxies
#proxy = collector.get_proxy({'code': 'us'})

# Retrieve only anonymous 'uk' or 'us' proxies
#proxy = collector.get_proxy({'code': ('us', 'uk'), 'anonymous': True})

# Retrieve all 'ca' proxies

#proxies = collector.get_proxies({'anonymous':True})

'''

#If you are copy pasting proxy ips, put in the list below
proxies = get_proxies()
'''
#proxy_pool = cycle(proxies)

from proxyscrape import create_collector
collector = create_collector('my-collector', 'http')


from pathlib import Path
import csv

data_dir = Path.cwd() / 'data' / 'news-outlets'
handle = "nytimes-2017"
print(handle)

loaded = []
with open(data_dir / "web-data-{}.csv".format(handle), 'r') as fin:
    reader = csv.reader(fin)
    for row in reader:
        loaded.append(row[0])

print(loaded)

with open(data_dir / "{}.csv".format(handle), "r") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row[0])
        if row[0] in loaded:
            print("{} Already loaded!".format(row[0]))
        else:
            print(row[0])
            payload = try_requesting(row)
            save_info(payload, row, data_dir, handle)

#        if payload:
#            tweet_id = row[0]
#            values = extract_metadata_nyt(payload, tweet_id)
#            with open(data_dir / "web-data-{}.csv".format(handle), "a") as fout:
#                writer = csv.writer(fout)
#                writer.writerow(values)
#        else:
##            with open(data_dir / "web-data-err.txt", 'a') as fout:
#                writer = csv.writer(fout)
#                writer.writerow()
