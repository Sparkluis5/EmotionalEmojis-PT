import sys
import tweepy
import csv

"""
Script for extracting tweets in real time with Tweepy and automatic labelling by analysing the emojis/hashtag in the
tweet. Use of this script is advised to be used in a virtual machine since its always running. All data is stored in
file dataset_raw.csv. Replace following variables with your own Tweepy credentials
"""

consumer_key= 'XYZ'
consumer_secret= 'XYZ'
access_key = 'XYZ'
access_secret = 'XYZ'


#Emoji/Emotion unicode variables definition

feliz = [u"\U0001F600", u"\U0001F602", u"\U0001F603", u"\U0001F604", u"\U0001F606", u"\U0001F607", u"\U0001F609",
         u"\U0001F60A", u"\U0001F60B", u"\U0001F60C", u"\U0001F60D",u"\u263A", u"\U0001F618", u"\U0001F61C",
         u"\U0001F61D", u"\U0001F61B", u"\u2764", u"\U0001F496", u"\U0001F495", u"\U0001F601", u"\u2665"]

irritado = [u"\U0001F62C", u"\U0001F620", u"\U0001F610", u"\U0001F611", u"\U0001F620", u"\U0001F621",
            u"\U0001F616", u"\U0001F624"]

nojo = [u"\U0001F4A9", u"\U0001F92E", u"\U0001F922"]

medo = [u"\U0001F605", u"\U0001F626", u"\U0001F627", u"\U0001F631", u"\U0001F628", u"\U0001F630"]

triste = [u"\U0001F614", u"\U0001F615", u"\u2639", u"\U0001F62B", u"\U0001F629", u"\U0001F622", u"\U0001F625",
          u"\U0001F62A" ,u"\U0001F613", u"\U0001F62D", u"\U0001F494"]

surpresa = [u"\U0001F633", u"\U0001F62F", u"\U0001F635", u"\U0001F632"]

queries = feliz + irritado + nojo + medo + triste + surpresa


#Tweepy API connection and authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

#Creation of custom stream listener, logic for data storage based on the text
class CustomStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        #Tweet with our query parameters
        print ('Text:' + status.author.screen_name, status.created_at, status.text, status.author.location, status.id)

        label = 'undified'
        tstring = status.text.replace('\n', ' ')

        #Label assignment
        for hemoji in feliz:
            if hemoji in status.text:
                label = 'Happiness'
        for iemoji in irritado:
            if iemoji in status.text:
                label = 'Anger'
        for nemoji in nojo:
            if nemoji in status.text:
                label = 'Disgust'
        for memoji in medo:
            if memoji in status.text:
                label = 'Fear'
        for temoji in triste:
            if temoji in status.text:
                label = 'Sadness'
        for semoji in surpresa:
            if semoji in status.text:
                label = 'Surprise'

        #Saving data to file
        with open('dataset_raw.csv', 'a', encoding='utf8') as f:
            writer = csv.writer(f)
            writer.writerow([status.author.screen_name, status.created_at, tstring, label, status.author.location, status.id])

    #Error and timeout handling
    def on_error(self, status_code):
        print  (sys.stderr, 'Encountered error with status code:', status_code)
        return True

    def on_timeout(self):
        print (sys.stderr, 'Timeout...')
        return True

#Custom streaming initialization, filter method uses query list for using emojis search
streamingAPI = tweepy.streaming.Stream(auth, CustomStreamListener())
streamingAPI.filter(track=queries, languages=["pt"])
