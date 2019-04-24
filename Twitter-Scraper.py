from secret import *
import tweepy
import time

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            time.sleep(15 * 60)

for status in limit_handled(tweepy.Cursor(api.search, q='k creisi',tweet_mode="extended").items(1)):
    print tweet

#tweets = api.search(q='k creisi', count=200, tweet_mode="extended")

