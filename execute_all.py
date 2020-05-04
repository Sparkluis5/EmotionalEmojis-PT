import os

#Script for executing crawler.py for all 6 Ekman basic emotions and all emojis. Currently not used since we use streaming_crawler-py

#auxliary variables
emotions = ["#irritado", "#nojo", "#medo", "#feliz", "#triste", "#surpresa"]
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

#Looping through emotions and emojis
for emotion in emotions:
    print("Executing Crawler for:" + emotion)
    os.system("python crawler.py " + emotion)
    print("Crawler Executed")

for emoji1 in feliz:
    print("Executing Crawler for:" + str(emoji1))
    os.system("python crawler.py " + str(emoji1))
    print("Crawler Executed")

for emoji2 in irritado:
    print("Executing Crawler for:" + str(emoji2))
    os.system("python crawler.py " + str(emoji2))
    print("Crawler Executed")

for emoji3 in nojo:
    print("Executing Crawler for:" + str(emoji3))
    os.system("python crawler.py " + str(emoji3))
    print("Crawler Executed")

for emoji4 in triste:
    print("Executing Crawler for:" + str(emoji4))
    os.system("python crawler.py " + str(emoji4))
    print("Crawler Executed")

for emoji5 in surpresa:
    print("Executing Crawler for:" + str(emoji5))
    os.system("python crawler.py " + str(emoji5))
    print("Crawler Executed")

for emoji6 in medo:
    print("Executing Crawler for:" + str(emoji6))
    os.system("python crawler.py " + str(emoji6))
    print("Crawler Executed")
