import os
import requests
from retrying import retry
from bs4 import BeautifulSoup
from pathlib import Path
import json
import csv
access_key = os.environ['NYT_ACCESS_KEY']
access_secret = os.environ['NYT_SECRET_KEY']

articlesearch_http = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'

database_directory = Path.cwd() / 'data' / 'news-outlets'


@retry(stop_max_attempt_number=5, wait_fixed=10000)
def fixed_rate_get(query):
    '''Load the URL and make sure to wait between requests.'''
    dat = requests.get(query)

    if response:
        return dat
    else:
        raise RuntimeError


def extract_metadata():
    '''Extract and save metadata'''
    dat = requests.get(query)
    soup = BeautifulSoup(dat)

    values = [dat.url]
    values.append(soup.find_all("meta",attrs={"property":"og:title"})[0].attrs['content'])
    values.append(soup.find_all("meta",attrs={"property":"twitter:title"})[0].attrs['content'])


def load_urls(data_dir=None, handle="nytimes"):
    with open(data_dir / "{}.csv".format(handle), "r") as f:
        reader = csv.reader(f)
        for row in reader:
            url = row[-1]
            print(url)
            #url = line.split(",")[-4]
            #soup = extract_metadata(url)
            with open("data/web-data-{}.csv".format(handle), "a") as fout:


                print('{}?fq=web_url:({})&api-key={}'.format(articlesearch_http, url, access_key))
                payload = requests.get('{}?q=:({})&api-key={}'.format(articlesearch_http, url, access_key))
                print(payload)
                print(payload.json())

            line = f.readline()

            return line


#test = load_urls(data_dir=database_directory, handle='nytimes-2017')

#url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json?q=election&api-key='+access_key
#r = requests.get(url)
#print(r.json())


q = "North Korea Says It Has Successfully Tested ICBM"

payload = requests.get('{}?q=:({})&begin_date=20170704&end_date=20170705&api-key={}'.format(articlesearch_http, q, access_key))
print(payload.json())
#load_urls(data_dir=database_directory,handle='nytimes-2017')
#print(r)
#json_data = r.json()
#print(json_data)

'''
def tag_attr_by_attr(soup, tag="meta", attrs={"property":"twitter:title"}, target_attr="content")

soup.find_all("meta",attrs={"property":"twitter:title"})[0].attrs['content']
'''
