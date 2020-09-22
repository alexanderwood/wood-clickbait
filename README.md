# See These Headline Writing Tips - Before Your Competitors Do!

Alt Titles:
* This Is How Buisiness Owners Are Earning Fortunes Writing Headlines!
* 10 Things You Need To Know Before You Share An Article

## The Problem

News articles are often shared to social media with a tagline that is different from the article headline on the news website itself. Sometimes, the social media tagline is simply a version of the full headline, shortened to meet character limits (e.g., 280 characters on Twitter). Other times, the headline is modified in order to attract more attention. 

Headlines that have been modified to read as misleading or controversial are referred to as -clickbait-. These headlines may misrepresent article contents in order to attract clicks. 

## Business Objectives

The business applications of these results are two-fold. 

In the first scenario is from the point of view of a news organization. By 2014, more than 2/3 of all domestic news revenue was generated by ad revenue [source]. Digital native news outlets, or news sites that are published exclusively online, **link sources about online ad revenue percentages**

Earn more money with more view counts by customizing the way social media outlines are written based on target demographics.  TODO

Outside of business applications, this information could also provide a deeper understanding of the clickbait problem for social science researchers. 

1. Create a database of Tweets and headlines from major news organizations.

2. Categorize headline-tagline pairs into subgroups.

3. Explore the relationship between headline type and click-through rate.  

4. Explore the relationship between headline type and article category.

5. Explore the relationship between headline type and 

## Data 

I will collect data on a variety of well-known news sources.

* New York Times
* 
* 
*
* 

### Headlines 

#### Tweets via Twitter API

* Time Range?
* All tweets? A random sample? (500,000 cap)


#### Linked Headlines via Web Scraper

Check web request time restrictions. How long will this take?

#### Feature Engineering

Several features will be taken from the tweets themselves: like count, retweet count, and the timestamp.

Additional features will be created using Natural Language Processing (NLP) techniques. First, I will examine metrics that can compare the similarity between the headline and the tweet tagline. 
* Edit Distance
* Jaccard Similarity 
* Smooth inverse frequency?
* Cosine sim

PreTrained Encoders
* InferSent by FB
* [Google Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4), a pretrained ML model which measures similarity with a syntatic

### Reliability Statistics

Reliability statistics are taken from several polls.

* Gallup

* Pew, April 2020; 2014

* The Economist, April 2018
 
 Compare these to headline types (regression?) - level of analysis depends on amount of reliability data available 


## Methods

### Feature Extraction

### Unsupervised Learning



## Maybe?

1. Train a model that takes an article headline as the input and yields a "clickbait"-style headline as an output. 
    * Question: What model? (Deep learning?) Is there a library that exists?
    * Question: 
1. Create additional features based on headline relevance to full article text. Question: How to create a headline relevance metric. 