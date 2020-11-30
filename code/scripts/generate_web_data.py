'''
Generate the web data from the hydrated & classified tweets.

Ideally this would run "online". Due to Twitter's Terms Of Use, I am only
permitted to share tweet IDs. Therefore, the data is not included in this
git repository. However, all the code required to reconstruct the datasets
on your own is available to you.
'''

# Load data.
from pathlib import Path
from datetime import datetime
import pandas as pd

# Load data & get the columns we need
fpath = Path.cwd() / 'code' / 'csv' / 'data_classified.csv'
df = pd.read_csv(fpath)
df['score'] = df.bait_proba.transform(lambda x: int(x>=0.5))
df['created_at'] = df.created_at.transform(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
df['month'] = df.created_at.transform(lambda x: x.month)
df['day'] = df.created_at.transform(lambda x: x.day)
df = df[['author_name', 'author_handle', 'month', 'day', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'score']]

# DAILY AGGREGATES
daily = df.groupby(['author_handle', 'month', 'day', 'score'])\
.agg({'retweet_count':['count', 'mean', 'sum'], 'reply_count':['mean', 'sum'],
      'like_count':['mean', 'sum'], 'quote_count':['mean', 'sum']})\
.sort_values(by=['author_handle', 'month', 'day'])\
.reset_index(level=[1,2,3])

daily.columns=['_'.join(col).strip() for col in daily.columns.values]
daily.rename(columns={'month_':'month',
                       'day_':'day',
                      'score_':'score',
                       'retweet_count_count':'count',
                       'retweet_count_mean': 'retweet_mean',
                       'retweet_count_sum': 'retweet_sum',
                       'reply_count_mean': 'reply_mean',
                       'reply_count_sum': 'reply_sum',
                       'like_count_mean': 'like_mean',
                       'like_count_sum': 'like_sum',
                       'quote_count_mean': 'quote_mean',
                       'quote_count_sum': 'quote_sum'},
             inplace=True)
daily.to_csv(Path.cwd() / 'code/csv/daily_aggregates.csv')
print(daily.head())

# MONTHLY AGGREGATES
monthly = df.groupby(['author_handle', 'month', 'score'])\
.agg({'retweet_count':['count', 'mean', 'sum'], 'reply_count':['mean', 'sum'],
      'like_count':['mean', 'sum'], 'quote_count':['mean', 'sum']})\
.sort_values(by=['author_handle', 'month'])\
.reset_index(level=[1,2])

monthly.columns=['_'.join(col).strip() for col in monthly.columns.values]
monthly.rename(columns={'month_':'month',
                      'score_':'score',
                       'retweet_count_count':'count',
                       'retweet_count_mean': 'retweet_mean',
                       'retweet_count_sum': 'retweet_sum',
                       'reply_count_mean': 'reply_mean',
                       'reply_count_sum': 'reply_sum',
                       'like_count_mean': 'like_mean',
                       'like_count_sum': 'like_sum',
                       'quote_count_mean': 'quote_mean',
                       'quote_count_sum': 'quote_sum'},
             inplace=True)
monthly.to_csv(Path.cwd()/ 'code/csv/monthly_aggregates.csv')
print(monthly.head())

# YEARLY AGGREGATES
yearly = df.groupby(['author_handle', 'score'])\
.agg({'retweet_count':['count', 'mean', 'sum'], 'reply_count':['mean', 'sum'],
      'like_count':['mean', 'sum'], 'quote_count':['mean', 'sum']})\
.sort_values(by=['author_handle'])\
.reset_index(level=[1])

yearly.columns=['_'.join(col).strip() for col in yearly.columns.values]
yearly.rename(columns={'score_':'score',
                       'retweet_count_count':'count',
                       'retweet_count_mean': 'retweet_mean',
                       'retweet_count_sum': 'retweet_sum',
                       'reply_count_mean': 'reply_mean',
                       'reply_count_sum': 'reply_sum',
                       'like_count_mean': 'like_mean',
                       'like_count_sum': 'like_sum',
                       'quote_count_mean': 'quote_mean',
                       'quote_count_sum': 'quote_sum'},
             inplace=True)
yearly.to_csv(Path.cwd() / 'code/csv/yearly_aggregates.csv')
print(yearly.head(20))
