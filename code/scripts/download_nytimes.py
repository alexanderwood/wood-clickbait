import os
import requests
from retrying import retry
from bs4 import BeautifulSoup

access_key = os.environ['NYT_ACCESS_KEY']
access_secret = os.environ['NYT_SECRET_KEY']

articlesearch_http = "https://api.nytimes.com/svc/search/v2/articlesearch.json"


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

def load_urls(handle="nytimes"):
    with open("data/tweet-data-{}.csv".format(handle), "r") as f:
        line = f.readline()
        while line:
            url = line.split(",")[-4]
            soup = extract_metadata(url)

            with open("data/web-data-{}.csv".format(handle), "a") as fout:
                values = [dat.url]



            line = f.readline()



'''
def tag_attr_by_attr(soup, tag="meta", attrs={"property":"twitter:title"}, target_attr="content")

soup.find_all("meta",attrs={"property":"twitter:title"})[0].attrs['content']
'''
