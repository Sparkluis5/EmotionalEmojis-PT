import re
import csv

d

#Auxliary variables definition, emojis and emotion definition
emo_list = [u"\U0001F600", u"\U0001F602",u"\U0001F603", u"\U0001F604", u"\U0001F606", u"\U0001F607", u"\U0001F609",
            u"\U0001F60A", u"\U0001F60B", u"\U0001F60C", u"\U0001F60D", u"\U0001F60E", u"\U0001F60F",
            u"\U0001F31E", u"\u263A", u"\U0001F618", u"\U0001F61C", u"\U0001F61D", u"\U0001F61B",
            u"\U0001F63A", u"\U0001F638", u"\U0001F639", u"\U0001F63B", u"\U0001F63C", u"\u2764",
            u"\U0001F496", u"\U0001F495", u"\U0001F601", u"\u2665", u"\U0001F62C", u"\U0001F620",
            u"\U0001F610", u"\U0001F611", u"\U0001F620", u"\U0001F621",
            u"\U0001F616", u"\U0001F624", u"\U0001F63E", u"\U0001F4A9", u"\U0001F605", u"\U0001F626",
            u"\U0001F627", u"\U0001F631", u"\U0001F628", u"\U0001F630", u"\U0001F640", u"\U0001F614",
            u"\U0001F615", u"\u2639", u"\U0001F62B", u"\U0001F629", u"\U0001F622", u"\U0001F625",
            u"\U0001F62A", u"\U0001F613", u"\U0001F62D", u"\U0001F63F", u"\U0001F4942", u"\U0001F633",
            u"\U0001F62F", u"\U0001F635", u"\U0001F632"]

emotions = ["#irritado", "#nojo", "#medo", "#feliz", "#triste", "#surpresa"]

feliz = [u"\U0001F600", u"\U0001F602", u"\U0001F603", u"\U0001F604", u"\U0001F606", u"\U0001F607", u"\U0001F609",
         u"\U0001F60A", u"\U0001F60B", u"\U0001F60C", u"\U0001F60D", u"\U0001F60E", u"\U0001F60F", u"\U0001F31E",
         u"\u263A", u"\U0001F618", u"\U0001F61C", u"\U0001F61D", u"\U0001F61B", u"\U0001F63A", u"\U0001F638",
         u"\U0001F639", u"\U0001F63B", u"\U0001F63C", u"\u2764", u"\U0001F496", u"\U0001F495", u"\U0001F601",
         u"\u2665"]
irritado = [u"\U0001F62C", u"\U0001F620", u"\U0001F610", u"\U0001F611", u"\U0001F620", u"\U0001F621",
            u"\U0001F616", u"\U0001F624", u"\U0001F63E"]
nojo = [u"\U0001F4A9"]
medo = [u"\U0001F605", u"\U0001F626", u"\U0001F627", u"\U0001F631", u"\U0001F628", u"\U0001F630", u"\U0001F640"]
triste = [u"\U0001F614", u"\U0001F615", u"\u2639", u"\U0001F62B", u"\U0001F629", u"\U0001F622", u"\U0001F625",
          u"\U0001F62A" ,u"\U0001F613", u"\U0001F62D", u"\U0001F63F", u"\U0001F4942"]
surpresa = [u"\U0001F633", u"\U0001F62F", u"\U0001F635", u"\U0001F632"]


#Find contradictory texts based on the existing emojis and hashtags, by finding if a text have two or more emojis that belong to different emotion classes
#Accepts a string corresponding to a text/tweet
def contradictory_tweet(text):
    #hashtag and emoji extraction from the given text
    hashs = get_hashtags(text)
    emojs = get_emojis(text)

    #finding if hashtags are different
    tot_hash = 0
    if len(hashs) > 1:
        for hashtag in hashs:
            if hashtag in emotions:
                tot_hash = tot_hash + 1

    #Finding if emojis presented belong to different subset
    if len(emojs) > 1:
        if(set(emojs).issubset(set(feliz)) or set(emojs).issubset(set(irritado)) or set(emojs).issubset(set(nojo))
           or set(emojs).issubset(set(medo)) or set(emojs).issubset(set(triste)) or  set(emojs).issubset(set(surpresa))):
            return False
        else:
            return True
    else:
        if tot_hash > 1:
            return True
        else:
            return False

#Simple function that accepts a string of a text/tweet and return all URL in input
def findURL(string):
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    return url

#Simple function that accepts a string of a text/tweet and return all emojis in input
def get_emojis(text):
    existing_emo = []
    for item in text.split():
        if item in emo_list:
            existing_emo.append(item)
    return existing_emo

#Simple function that accepts a string of a text/tweet and return all hashtags in input
def get_hashtags(text):
    tags = set([item.strip(".,-\"\'&*^!") for item in text.split() if (item.startswith("#") and len(item) < 256)])
    return sorted(tags)

#Simple function that accepts a string of a text/tweet and removes all URLs in input and returns the resulting string
def strip_links(text):
    result = re.sub(r"http\S+", "@URL", text)
    return result

#Simple function that accepts a string of a text/tweet and removes all URLs and Username mentions in input and returns the resulting string
def strip_all_entities(text):
    retstring = re.sub('@[^\s]+', '@USERNAME', text)
    retstring = re.sub('#\S+', '', retstring)
    return retstring

#Script for processing all extracted texts, remove contradictory, small and duplicates texts. Creates new file with cleaned data
def processData():
    with open('data.csv', 'r', encoding='utf8') as in_file, open('data_cleaned.csv', 'w', encoding='utf8') as out_file:
        reader = csv.reader(in_file)
        writer = csv.writer(out_file)

        #Auxiliary variables
        seen = set()
        i = 1
        removed_duplicates = 0
        removed_contraditory = 0
        removed_small = 0
        removed_url = 0
        loc = ''

        #Looping through the raw data file
        for row in csv.reader((line.replace('\0', '') for line in in_file), delimiter=","):
            #ignore bad file format
            if len(row) == 1:
                continue
            if len(row) < 1:
                continue

            #Ingoring duplicates that might persists
            temp = ''.join(row[2])
            if temp in seen:
                removed_duplicates = removed_duplicates + 1
                continue
            else:
                seen.add(temp)
                if len(row[2].split()) < 3:
                    removed_small = removed_small + 1
                    continue
                if contradictory_tweet(temp):
                    removed_contraditory = removed_contraditory + 1
                    continue
                #ignore bad labelled texts
                if row[3] == 'undified':
                    continue
                transformed_line = strip_links(strip_all_entities(str(row[2])))
                transformed_line = transformed_line.replace('\n', ' ')
                #Write in new file
                writer.writerow([row[0], row[1], transformed_line, row[3], row[4]])
                i = i + 1

        #Report of data transformations performed
        print("--REPORT--")
        print("Duplicate Rows Removed:" + str(removed_duplicates))
        print("Small Tweets Removed:" + str(removed_small))
        print("Contraditory Tweets Removed:" + str(removed_contraditory))
        print("Tweets with URLS Removed:" + str(removed_url))
        print("Total Occurances:" + str(float(i) / float(2)))


if __name__ == '__main__':
    processData()