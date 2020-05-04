import tweepy
import csv
import sys

#Script for text and emotion assignment of Twitter texts on last week due to API limit

#Emoji/emotion variables definition, we used unicode to define each emoji

feliz = [[u"\U0001F600"], [u"\U0001F602"], [u"\U0001F603"], [u"\U0001F604"], [u"\U0001F606"], [u"\U0001F607"], [u"\U0001F609"],
         [u"\U0001F60A"], [u"\U0001F60B"], [u"\U0001F60C"], [u"\U0001F60D"], [u"\U0001F60E"], [u"\U0001F60F"], [u"\U0001F31E"],
         [u"\u263A"], [u"\U0001F618"], [u"\U0001F61C"], [u"\U0001F61D"], [u"\U0001F61B"], [u"\U0001F63A"], [u"\U0001F638"],
         [u"\U0001F639"], [u"\U0001F63B"], [u"\U0001F63C"], [u"\u2764"], [u"\U0001F496"], [u"\U0001F495"], [u"\U0001F601"],
         [u"\u2665"]]
irritado = [[u"\U0001F62C"], [u"\U0001F620"], [u"\U0001F610"], [u"\U0001F611"], [u"\U0001F620"], [u"\U0001F621"],
            [u"\U0001F616"], [u"\U0001F624"], [u"\U0001F63E"]]
nojo = [[u"\U0001F4A9"]]
medo = [[u"\U0001F605"], [u"\U0001F626"], [u"\U0001F627"], [u"\U0001F631"], [u"\U0001F628"], [u"\U0001F630"], [u"\U0001F640"]]
triste = [[u"\U0001F614"], [u"\U0001F615"], [u"\u2639"], [u"\U0001F62B"], [u"\U0001F629"], [u"\U0001F622"], [u"\U0001F625"],
          [u"\U0001F62A"] ,[u"\U0001F613"], [u"\U0001F62D"], [u"\U0001F63F"], [u"\U0001F4942"]]
surpresa = [[u"\U0001F633"], [u"\U0001F62F"], [u"\U0001F635"], [u"\U0001F632"]]

#Tweepy variables definition, assign your own
consumer_key = 'XYZ'
consumer_secret = 'XYZ'
access_token = 'XYZ'
access_token_secret = 'XYZ'

#Tweppy initialization
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

print("-- Crawler Starting --")

if len(sys.argv) < 1:
    print("No emotion associated with crawler")
    exit()

#emotion serach based on script args
emotion = sys.argv[1]

print("Starting tweets with:" + emotion)

#Write data to specific emotion file
if emotion[0] == '#':
    filename = emotion[1:] + ".csv"
else:
    for emo1 in feliz:
        if emotion == str(emo1):
            filename = "happiness.csv"
    for emo2 in irritado:
        if emotion == str(emo2):
            filename = "anger.csv"
    for emo3 in nojo:
        if emotion == str(emo3):
            filename = "disgust.csv"
    for emo4 in medo:
        if emotion == str(emo4):
            filename = "fear.csv"
    for emo5 in triste:
        if emotion == str(emo5):
            filename = "sadness.csv"
    for emo6 in surpresa:
        if emotion == str(emo6):
            filename = "surprise.csv"

csvFile = open(filename, 'a')
csvWriter = csv.writer(csvFile)

print("Starting Extraction")

#Looping though available tweets that correspond to query search
for tweet in tweepy.Cursor(api.search,q=emotion, lang="pt").items(3000):
    print(tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])


