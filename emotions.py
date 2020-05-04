import pandas as pd
import csv, re, regex, emoji, string

"""
Script for dataset transformation to produce a new dataset where the texts are the ones with only one emoji and are 
placed at the end of the text/tweet. This produces a new dataset where the emotion/emoji relation is more stronger than
the original dataset. This script was developed for a new set of experiments.
"""


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

all_emojis = feliz + irritado + nojo + medo + triste + surpresa

#Auxiliary function to remove URL from a given string, returns cleaned string
def strip_links(text):
    result = re.sub(r"http\S+", " ", text)
    return result

#Auxiliary function to remove Twitter mentions and Hashtags from a given string, returns cleaned string
def strip_all_entities(text):
    retstring = re.sub('@[^\s]+', ' ', text)
    retstring = re.sub('#\S+', '', retstring)
    return retstring

#Auxiliary function to remove Retweet token from a given string, returns cleaned string
def remove_rt(text):
    remrt = text.replace('RT @USERNAME', '')
    remrt = " ".join(remrt.split())
    return remrt

#Auxiliary function to remove punctuation from a given string, returns cleaned string
def remove_punc(text):
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    text = ''.join([i for i in text if not i.isdigit()])
    return text

#Auxiliary function to count total ammount of emojis and words from a given string, returns emoji and word counts and list of existing emojis
def split_count(text):
    emo_list = []
    emoji_counter = 0
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in all_emojis for char in word):
            emoji_counter += 1
            # Remove from the given text the emojis
            emo_list.append(word)
            text = text.replace(word, '')

    words_counter = len(text.split())

    return emoji_counter, words_counter,emo_list


#Function to remove emojis from a given text. Accepts a string consisting of a text a returns cleaned string without emojis
def clear_emoji(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])

    return clean_text

#Function for dealing with dataset by creating a new where all texts have just one emoji at the end. Accepts a string consisting of a filename
def lastText(filename):
    end_emo = 0

    header = ['Username', 'Data', 'Tweet Original', 'Tweet Limpo', 'Emocao', 'Localizacao', 'Emoji', 'NRCValence'
        , 'NRCArousal', 'NRCDominance', 'ANEWValence'
        , 'ANEWArousal', 'ANEWDominance', 'Sentiment', 'Negacoes', 'Total Palavras', 'Maior Palavra',
              'Media de Carateres'
        , 'Pontos de Exclamacao', 'Palavras Maiusculas', 'Ponto de Interrogacao']

    all_data = pd.read_csv(filename, names=header, skiprows=1)
    with open('dataset_endemoji.csv', 'a', encoding='utf8') as f:
        writer = csv.writer(f)
        for index, row in all_data.iterrows():
            tstring = row['Tweet Original']
            word_list = tstring.split()
            if(word_list[-1] in all_emojis):
                end_emo += 1
                tstring = remove_rt(tstring)
                tstring = strip_links(tstring)
                tstring = strip_all_entities(tstring)
                tstring = clear_emoji(tstring)
                writer.writerow([row['Username'], row['Data'], row['Tweet Original'], tstring, row['Emocao'],
                                 row['Localizacao'], row['Emoji'], row['NRCValence'], row['NRCArousal'],
                                 row['NRCDominance'],
                                 row['ANEWValence'], row['ANEWArousal'], row['ANEWDominance'], row['Sentiment'],
                                 row['Negacoes'], row['Total Palavras'], row['Maior Palavra'],
                                 row['Media de Carateres'],
                                 row['Pontos de Exclamacao'], row['Palavras Maiusculas'], row['Ponto de Interrogacao']])




if __name__ == '__main__':
    #dataCleanse()
    #lastText()
    pass