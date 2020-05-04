# coding=utf-8

import csv
import pandas as pd
import statistics
import re
import string
import numpy as np
import gensim
import nltk
import logging
import pickle
import keras.backend as K

from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from gensim.models.callbacks import CallbackAny2Vec
from stanfordcorenlp import StanfordCoreNLP
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import chi2
from tqdm import tqdm
from nltk import tokenize
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.cluster import KMeansClusterer
from nltk.stem.wordnet import WordNetLemmatizer
from spacy.lang.pt.lemmatizer import LOOKUP
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from datetime import datetime

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

"""
Script composing of model creation and data processing for emotion classification from text. Each function has its own description
and several methods needs better optimization.
"""

#Classes necessary for data pipeline to apply to develped models
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]


class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1

#Path for stanford-corenlp, its necessary for data processing
local_corenlp_path = '/Users/luisduarte/UC/stanford-corenlp-full-2018-10-05'

#lexicons paths and stopwords definition
stops = set(stopwords.words("portuguese"))
nrc = '/Lexicons/NRC-PT.txt'
path = '/Lexicons/LEED.csv'
lmtzr = WordNetLemmatizer()
spcnlp = StanfordCoreNLP(local_corenlp_path)

#Auxiliary functions, mainly dataframes headers
vars = ['Word', 'Wordpt', 'Valence', 'Arousal', 'Dominance']
sent_vars = ['Word','Flex','Tg','Pol','Anot']
anew_vars = ["Number","E-Word","EP-Word","Val-M","Val-SD","Arou-M","Arou-SD","Dom-M","Dom-SD","Freq","Nlett","Nsyll","GClass","Neigh"]
anew_data = pd.read_csv('/Lexicons/ANEW_PT.csv',skiprows=1, sep=';', names=anew_vars)

#read lexicons
lex_data = pd.read_csv(nrc, sep='\t', names=vars)
senti_data = pd.read_csv('/Lexicons/Sentilex.txt', sep=';', names=sent_vars)
temp = senti_data['Word'].str.split(",", n = 1, expand = True)
senti_data['Word'] = temp[0]
temp = senti_data['Pol'].str.split("=", n = 1, expand = True)
senti_data['Pol'] = temp[1]


#Emojis list definition
emo_list = [u"\U0001F600", u"\U0001F602",u"\U0001F603", u"\U0001F604", u"\U0001F606", u"\U0001F607", u"\U0001F609",
            u"\U0001F60A", u"\U0001F60B", u"\U0001F60C", u"\U0001F60D", u"\U0001F60E", u"\U0001F60F",
            u"\U0001F31E", u"\u263A", u"\U0001F618", u"\U0001F61C", u"\U0001F61D", u"\U0001F61B",
            u"\U0001F63A", u"\U0001F638", u"\U0001F639", u"\U0001F63B", u"\U0001F63C", u"\u2764",
            u"\U0001F496", u"\U0001F495", u"\U0001F601", u"\u2665", u"\U0001F62C", u"\U0001F620",
            u"\U0001F610", u"\U0001F611", u"\U0001F620", u"\U0001F621",
            u"\U0001F616", u"\U0001F624", u"\U0001F63E", u"\U0001F4A9", u"\U0001F605", u"\U0001F626",
            u"\U0001F627", u"\U0001F631", u"\U0001F628", u"\U0001F630", u"\U0001F640", u"\U0001F614",
            u"\U0001F615", u"\u2639", u"\U0001F62B", u"\U0001F629", u"\U0001F622", u"\U0001F625",
            u"\U0001F62A", u"\U0001F613", u"\U0001F62D", u"\U0001F63F", u"\U0001F494", u"\U0001F633",
            u"\U0001F62F", u"\U0001F635", u"\U0001F632"]

def get_lemma(text):
    return LOOKUP.get(text, text)


#Transformation of elongated words in reduced forms, accepts a string a returns a new string with continuous letters simplified
def transform_elongated_word(text):
    new_text = re.sub(r'(.)\1{3,}', r'\1', text)
    return new_text

#Function to check of string is elongated (eg:hellooooo) returns 1 or 0
def has_long(text):
    elong = re.compile("([a-zA-Z])\\1{3,}")
    val = bool(elong.search(text))
    if val == True:
        ret = 1
    else:
        ret = 0
    return ret

#Function to check the ammount of uppercased words in a given string (text/tweet) withou username or url uniform tokens
def count_uppercase_words(text):
    check = text.split()
    if check.count('@USERNAME') > 0:
        check.remove('@USERNAME')
    if check.count('@URL') > 0:
        check.remove('@URL')
    upper_words = sum(map(str.isupper, check))
    return upper_words


#Function to count the ammount of exclamation points of a given text
def count_exclamation_letter(text):
    total_exclamation = text.count('!')
    return total_exclamation


#Function to count the ammount of interrogation points of a given text
def count_interrogation_letter(text):
    total_interrogation = text.count('?')
    return total_interrogation

#Function to count the total ammount of words of a given text
def count_word_length(text):
    wordscount = []
    totalwords = len(text.split())
    for word in text.split():
        wordscount.append(len(word))
    maxchar = max(wordscount)
    avgchar = np.mean(wordscount)
    return totalwords, maxchar, round(avgchar)

#Function to count the ammount of negated words of a given text
def count_neg_words(text):
    negwords = ['não', 'nao', 'nem', 'nunca', 'jamais', 'nenhum']
    num_neg = 0
    for word in text.split():
        if word.lower() in negwords:
            num_neg = num_neg + 1
    return num_neg

#Function to transform a given text in a list of n-grams. Accepts a string corresponding to a text, a int n for the n-gram value. Returns list
def get_ngrams(text, n):
    list_ngram = []
    ngram = ngrams(text.split(), n)
    for g in ngram:
        list_ngram.append(g)

    return list_ngram


#Function to remove strange chars that cause errors in latter stages
def removal_chars(sentences):
    sentence_list = []
    count = 0
    for sen in sentences:
        count += 1
        s = nltk.word_tokenize(sen)
        characters = ["á", "\xc3", "\xa1", "\n", ",", "."]
        new = ' '.join([i for i in s if not [e for e in characters if e in i]])
        sentence_list.append(new)
    return sentence_list

#Auxiliary function
def simplify_tag(t):
    if "+" in t:
        return t[t.index("+")+1:]
    else:
        return t

#Function to read dataset, accepts a string corresponding to the desired dataset and boolean cats for removing cat emojis
def read_data(filename,cats):
    header = ['Username', 'Data', 'Tweet Original', 'Tweet Limpo', 'Emocao', 'Localizacao', 'Emoji', 'NRCValence'
        , 'NRCArousal', 'NRCDominance', 'ANEWValence'
        , 'ANEWArousal', 'ANEWDominance', 'Sentiment', 'Negacoes', 'Total Palavras', 'Maior Palavra',
              'Media de Carateres'
        , 'Pontos de Exclamacao', 'Palavras Maiusculas', 'Pontos de Interrogacao']

    print('Reading dataset...')
    all_data = pd.read_csv('/datasets/' + filename +'.csv', names=header, skiprows=1)
    all_data = all_data[all_data['Emoji'] != 'None']
    all_data['Tweet Limpo'] = all_data['Tweet Limpo'].values.astype('U')
    all_data['Emoji'] = all_data['Emoji'].values.astype('U')

    if cats:
        all_data = selectData(True)

    return all_data

#Create traditional model function
def create_model(filename,model,ngram_inf,ngram_sup,label,cross,cats,tSize,news_data,save,sampler):
    all_data = read_data(filename,cats)

    #Train and test split
    train, test = train_test_split(all_data, random_state=42, test_size=tSize, shuffle=True)

    print('Creating Pipeline')

    if sampler == 'Oversampler':
        text = make_pipeline(
            CountVectorizer(stop_words=stopwords.words('portuguese'), ngram_range=(ngram_inf, ngram_sup)),
            RandomOverSampler())

    if sampler == 'Undersampler':
        text = make_pipeline(
            CountVectorizer(stop_words=stopwords.words('portuguese'), ngram_range=(ngram_inf, ngram_sup)),
            RandomUnderSampler())

    if sampler == None:
        text = Pipeline([
            ('selector', TextSelector(key='Tweet Limpo')),
            ('count', CountVectorizer(stop_words=stopwords.words('portuguese'), ngram_range=(ngram_inf,ngram_sup)))])

    #Additional features extracted from all texts, currently not used since it introduces more confusion to our models
    neg = Pipeline([('feat', NumberSelector(key='Negacoes')), ])
    val = Pipeline([('feat', NumberSelector(key='ANEWValence')), ])
    aro = Pipeline([('feat', NumberSelector(key='ANEWArousal')), ])
    dom = Pipeline([('feat', NumberSelector(key='ANEWDominance')), ])
    numbers = Pipeline([('feat', NumberSelector(key='Total Palavras')), ])
    word_len = Pipeline([('feat', NumberSelector(key='Maior Palavra')), ])
    avg_len = Pipeline([('feat', NumberSelector(key='Media de Carateres')), ])
    exc_count = Pipeline([('feat', NumberSelector(key='Pontos de Exclamacao')), ])
    upper_count = Pipeline([('feat', NumberSelector(key='Palavras Maiusculas')), ])
    int_count = Pipeline([('feat', NumberSelector(key='Pontos de Interrogacao')), ])

    #Feature Union, in case we want to add the features
    print('Unindo as features')

    feats = FeatureUnion([
                          ('text', text),
                          ])

    #Model selection
    if model == 'NB':
        pipeline = Pipeline([('features', feats),('clf', MultinomialNB(fit_prior=True)),])
    if model == 'SVM':
        pipeline = Pipeline([(('features', feats)),
                             ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=400, tol=None, verbose=True),)])

    #Task definition, assign Y label, emotions or emojis
    if label == 'Emocao':
        classes = ["Felicidade", "Raiva", "Nojo", "Medo", "Surpresa", "Tristeza"]
    else:
        classes = all_data[label].unique()


    #10 Cross Validation action, saves model and results variables if defined
    if cross:
        pipeline.fit(all_data, all_data[label])
        accuNB = cross_val_score(pipeline, all_data, all_data[label], cv=10, scoring='precision_weighted')

        print('Cross Validation accuracy mean is {}'.format(accuNB.mean()) + str(' for model: ') + model)
        print('Cross Validation accuracy std is {}'.format(accuNB.std()) + str(' for model: ') + model)

        if save:
            #Hardcoded filename, gets hard to distinguish each models, this ways its more easy at latter steps
            saved_filename = model + '_CV10_' + str(datetime.now().day) + '_' + str(datetime.now().hour) + '_' + str(datetime.now().minute) + '.p'
            pickle.dump([pipeline,accuNB], open(saved_filename, "wb"))

    #Simple train and test validation, saves model and additional variables if defined
    else:
        pipeline.fit(train, train[label])
        accuNB = pipeline.predict(test)
        cm, conf_mat, report = plot_confusion_matrix(test[label], accuNB, classes)

        if save:
            saved_filename = model + '_Simple_' + str(datetime.now().day) + '_' + str(datetime.now().hour) + '_' + str(datetime.now().minute) + '.p'
            pickle.dump([pipeline,accuNB, conf_mat, report], open(saved_filename, "wb"))

    #Perform validation with news dataset
    if news_data:
        new_data = newDataset()
        acc = pipeline.predict(new_data)
        print('Test accuracy is {}'.format(accuracy_score(new_data['Emocao'], acc)))
        cm, conf_mat, report = plot_confusion_matrix(new_data['Emocao'], acc, classes)

        if save:
            saved_filename = model + '_News_' + str(datetime.now().day) + '_' + str(datetime.now().hour) + '_' + str(datetime.now().minute) + '.p'
            pickle.dump([acc, conf_mat, report], open(saved_filename, "wb"))

#Auxiliary function for recall classification, used in neural network training
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

#Auxiliary function for precision classification, used in neural network training
def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

#Auxiliary function for F1 Score classification, used in neural network training
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def create_lstm(label,epochs,filename,sampler,tSize,save,news_data):
    all_data = read_data(filename, False)

    #Y label assignment based on our task
    if label == 'Emocao':
        classes = ["Felicidade", "Raiva", "Nojo", "Medo", "Surpresa", "Tristeza"]
        dense_val = 6
    else:
        classes = emo_list
        dense_val = 62

    #label encoder definition, needed for passing our text label to a numerical form
    le = preprocessing.LabelEncoder()

    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 150000
    # Max number of words in each tweet.
    MAX_SEQUENCE_LENGTH = 20
    EMBEDDING_DIM = 100

    #Tokenizer definition
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(all_data['Tweet Limpo'])

    X = tokenizer.texts_to_sequences(all_data['Tweet Limpo'])
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

    print('Shape of data tensor:', X.shape)
    Y = all_data[label]

    #Sampling dataset if required
    if sampler == 'Undersampler':
        rus = RandomUnderSampler(return_indices=True)
        X, Y, idx_resampled = rus.fit_sample(X, Y)

    if sampler == 'Oversampler':
        rus = RandomOverSampler(return_indices=True)
        X, Y, idx_resampled = rus.fit_sample(X, Y)

    #Train and test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=tSize, random_state=42)
    Y_train = pd.get_dummies(Y_train)

    #Model definition, easly changable
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(dense_val, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc', f1_m, precision_m, recall_m])
    print(model.summary())

    epochs = epochs
    batch_size = 64

    #Model fit
    le.fit(Y_test)
    fit_labels = le.transform(Y_test)
    history = model.fit(X_train, pd.get_dummies(Y_train), epochs=epochs, batch_size=batch_size, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    Y_pred = model.predict_classes(X_test)

    #plot_history(history)
    cm, conf_mat, report = plot_confusion_matrix(fit_labels, Y_pred, classes)
    print(report)

    if save == True:
        saved_filename = 'LSTM_' + sampler + '_' + str(datetime.now().day) + '_' + str(datetime.now().hour) + '_' + str(
            datetime.now().minute) + '.p'
        pickle.dump([model, tokenizer, conf_mat, report], open(saved_filename, "wb"))

    if news_data == True:
        #Data processing
        new_data = newDataset()
        X_new = tokenizer.texts_to_sequences(new_data['Tweet Limpo'])
        X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)
        Y_new = new_data[label]
        Y_new2 = new_data[label]
        Y_new = pd.get_dummies(Y_new)

        #Applying data to our model
        Y_pred_new = model.predict_classes(X_new)
        le.fit(Y_new2)
        fit_labels_new = le.transform(Y_new2)
        cm, conf_mat, report = plot_confusion_matrix(fit_labels, Y_pred, classes)
        print(report)

        if save == True:
            saved_filename = 'LSTM_News_' + sampler + '_' + str(datetime.now().day) + '_' + str(
                datetime.now().hour) + '_' + str(
                datetime.now().minute) + '.p'
            pickle.dump([model, tokenizer, conf_mat, report], open(saved_filename, "wb"))


"""
Function to plot the confusion matrix of our results. 
y_true - A list corresponding to the true labels of our tests
y_pred - A list corresponding to the obtained labels of the developed models, must be the same length as y_true
classes - a list corresponding where each element corresponds to a string of the name of a class
"""
def plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    #Plot initialization
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix', ylabel='Real Label', xlabel='Predicted Label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

    #Get classification report, obtains f1, precision, recall values. Can be adapted
    report = classification_report_imbalanced(y_true, y_pred, target_names=classes)
    print(report)

    return ax,cm,report

#Function to plot learning curve.
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

#Fuction to plot model training history, available only for the neural network model
def plot_history(history):
    #Obtaining history values
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    #Bad training cases, ignores this function
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    # As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    # Plotting Losses
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    #Plotting the Accuracy values
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

#Creates sparse Matrix of TF e TFIDF of a dataset. Needs to define the inferior and superior n-grams range. file_saved is a flag for saving output to file
def showSparseMatrix(filename,ngram_inf, ngram_sup,max_feat, file_saved):
    all_data = read_data(filename, False)

    print("Calculating TF...")

    termVectorizer = CountVectorizer(stop_words=stopwords.words('portuguese'), ngram_range=(ngram_inf, ngram_sup), max_features=max_feat)
    tfidfVectorizer = TfidfVectorizer(stop_words=stopwords.words('portuguese'), ngram_range=(ngram_inf, ngram_sup), max_features=max_feat)

    termData = termVectorizer.fit_transform(all_data['Tweet Limpo'])
    tfidfData = tfidfVectorizer.fit_transform(all_data['Tweet Limpo'])

    TF_names = ['Tweet {:d}'.format(idx) for idx, _ in enumerate(termData)]
    TFIDF_names = ['Tweet {:d}'.format(idx) for idx, _ in enumerate(tfidfData)]
    sparseTF_Matrix = pd.DataFrame(data=termData.toarray(), index=TF_names, columns=termVectorizer.get_feature_names())
    sparseTFIDF_Matrix = pd.DataFrame(data=tfidfData.toarray(), index=TFIDF_names, columns=tfidfVectorizer.get_feature_names())

    print(sparseTF_Matrix)
    print(sparseTFIDF_Matrix)

    print("Calcular Chi2...")
    chi2score = chi2(termData, all_data['Emocao'])[0]
    wscores = zip(termVectorizer.get_feature_names(), chi2score)
    wchi2 = sorted(wscores, key=lambda x: x[1])

    print(wchi2)

    data_class = ['feliz', 'medo', 'irritado', 'triste', 'surpresa', 'nojo']
    N = 10

    print("Calcular Features...")

    for Emotion in sorted(data_class):
        features_chi2 = chi2(termData.toarray(), all_data['Emocao'] == Emotion)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(termVectorizer.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        triigrams = [v for v in feature_names if len(v.split(' ')) == 3]
        print("# '{}':".format(Emotion))
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
        print("  . Most correlated trigrams:\n. {}".format('\n. '.join(triigrams[-N:])))

    # Saving matrixes and chi2 values to file
    if file_saved == 1:
        saved_filename = 'SparseMatrixes_' + filename + '.p'
        pickle.dump([sparseTF_Matrix, sparseTFIDF_Matrix, wchi2], open(saved_filename, "wb"))

    return sparseTF_Matrix, sparseTFIDF_Matrix, wchi2



"""
Function to process a given text to obtain several Valence, Arousal and Dominance values from several lexicons, available in folder /Lexicons
We use NRC Lexicon, Anew-PT, Sentilex for obtaining sentiment. We match each word from the input and procede to find its occurance in the lexicons,
where if it happens we obtain the values that are available in the lexicon. After looping through all words we perform a mean operation on the obtained values
and verify if there is a predominant sentiment (based on the occurance). We also perfomed inverse operation if we find negated words. All data is then saved to file. 
"""
def analyzefile(fulltext):
    #Sentence tokenization and auxiliary variables
    sentences = tokenize.sent_tokenize(fulltext)
    tot_val = []
    tot_aro = []
    tot_dom = []
    tot_senti = []
    valenceNRC = 0.0
    arousalNRC = 0.0
    dominanceNRC = 0.0
    valenceANEW = 0.0
    arousalANEW = 0.0
    dominanceANEW = 0.0

    #Looping through all sentences
    for s in sentences:
        #Auxiliary values for sentence analysis
        all_words = []
        found_words = []
        total_words = 0
        v_list = []
        a_list = []
        d_list = []
        v_nrc_list = []  # Valence of NRC
        a_nrc_list = []  # Arousal of NRC
        d_nrc_list = []  # Dominance of NRC
        v_anew_list = []  # Valence of ANEW
        a_anew_list = []  # Arousal of ANEW
        d_anew_list = []  # Dominance of ANEW
        sentiment_list = [] # Sentimento values of Sentilex

        # PoS Tagging in Portuguese
        doc = spcnlp(s.lower())
        words = []

        for tokens in doc:
            tuple = (tokens.text, tokens.pos_)
            words.append(tuple)

        for index, p in enumerate(words):
            # Ignoring stopwords
            w = p[0]
            pos = p[1]
            if w in stops:
                continue

            #Verifying negated forms
            j = index - 1
            neg = False
            while j >= 0 and j >= index - 3:
                if words[j][0] == 'não' or words[j][0] == 'nem':
                    neg = True
                    break
                j -= 1

            lemma = w

            # Needs better lematizer for portuguese, not available right now
            #if pos[0] == 'NOUN' or pos[0] == 'VERB':
            #    lemma = lmtzr.lemmatize(w, pos=pos[0].lower())
            #else:
            #   lemma = w

            all_words.append(lemma)

            # search for values in NRC lexicon
            leed_data = open(path, 'r', encoding='utf-8', errors='ignore')
            check = lex_data.loc[lex_data['Wordpt'] == lemma.casefold(), ['Valence', 'Arousal', 'Dominance']]

            # Word found in lexicon, negated forms are treated by inverting obtained values
            if len(check) > 0:
                if neg:
                    found_words.append("neg-" + lemma)
                else:
                    found_words.append(lemma)
                v = float(check.iloc[0]['Valence'])
                a = float(check.iloc[0]['Arousal'])
                d = float(check.iloc[0]['Dominance'])

                if neg:
                    # Reverting polarity
                    v = 5 - (v - 0.5)
                    a = 5 - (a - 0.5)
                    d = 5 - (d - 0.5)

                v_nrc_list.append(v)
                a_nrc_list.append(a)
                d_nrc_list.append(d)

            # search for values in ANEW-PT lexicon
            check = anew_data.loc[anew_data['EP-Word'] == lemma.casefold()]

            #Word found in lexicon, negated forms are treated by inverting obtained values
            if len(check) > 0:
                if neg:
                    found_words.append("neg-" + lemma)
                else:
                    found_words.append(lemma)
                v = float(check.iloc[0]['Val-M'].replace(',', '.'))
                a = float(check.iloc[0]['Arou-M'].replace(',', '.'))
                d = float(check.iloc[0]['Dom-M'].replace(',', '.'))

                if neg:
                    # Reverting polarity
                    v = 5 - (v - 0.5)
                    a = 5 - (a - 0.5)
                    d = 5 - (d - 0.5)

                v_anew_list.append(v)
                a_anew_list.append(a)
                d_anew_list.append(d)

            # Check Sentilex lexicon for word sentiment, negated words are inversed
            check = senti_data.loc[senti_data['Word'] == lemma.casefold()]
            if len(check) > 0:
                if neg:
                    found_words.append("neg-" + lemma)
                else:
                    found_words.append(lemma)
                sentiment_val = int(check.iloc[0]['Pol'])
                sentiment_list.append(sentiment_val)

            #Searchs in LEED lexicon
            for row in csv.reader(leed_data, delimiter=';'):
                if lemma.encode('unicode_escape') in row[0].encode('unicode_escape'):
                    v = float(row[4].replace(',', '.'))
                    a = float(row[5].replace(',', '.'))
                    d = 0.0
                    v_list.append(v)
                    a_list.append(a)
                    d_list.append(d)
                    tot_val.append(v)
                    tot_aro.append(a)
                    tot_dom.append(d)
                    break

    #Verifying the obtained values of sentiment of the input text. Assigns the most frequent sentiment
    if len(sentiment_list) > 0:
        pos = sentiment_list.count(1)
        neg = sentiment_list.count(-1)
        if pos == neg:
            tot_senti = 0
        else:
            if pos > neg:
                tot_senti = 1
            else:
                tot_senti = -1
    else:
        tot_senti = 0

    # Verifying the obtained values of Valence, Arousal and Dominance of the input text of the lexicons. Assigns the mean of the obtained values
    if len(v_nrc_list) == 0:
        valenceNRC = 0.0
        arousalNRC = 0.0
        dominanceNRC = 0.0
    else:  # output sentiment info for this sentence
        valenceNRC = statistics.mean(v_nrc_list)
        arousalNRC = statistics.mean(a_nrc_list)
        dominanceNRC = statistics.mean(d_nrc_list)
        valenceNRC = float(format(valenceNRC, '.4f'))
        arousalNRC = float(format(arousalNRC, '.4f'))
        dominanceNRC = float(format(dominanceNRC, '.4f'))
        if len(v_anew_list) == 0:
            valenceANEW = 0.0
            arousalANEW = 0.0
            dominanceANEW = 0.0
        else:
            valenceANEW = statistics.mean(v_anew_list)
            arousalANEW = statistics.mean(a_anew_list)
            dominanceANEW = statistics.mean(d_anew_list)
            valenceANEW = float(format(valenceANEW, '.4f'))
            arousalANEW = float(format(arousalANEW, '.4f'))
            dominanceANEW = float(format(dominanceANEW, '.4f'))

    return valenceNRC, arousalNRC, dominanceNRC, valenceANEW, arousalANEW, dominanceANEW, tot_senti



#Creates a gensim word2vec model of a given dataset, saves to file
def create_wordvec(filename):
    all_data = read_data(filename, False)
    tmp_corpus = all_data['Tweet Original'].map(lambda x: x.split('.'))

    corpus = []

    #Text formating, necessary for not producing bad results
    for num in tqdm(range(1,len(tmp_corpus))):
        for line in tmp_corpus[num]:
            line = line.translate(str.maketrans('', '', string.punctuation))
            line.replace('RT', '@RT')
            line.replace('USERNAME','@USERNAME')
            line.replace('URL', '@URL')
            line = transform_elongated_word(line)
            line = line.lower()

            words = [x for x in line.split()]
            temp_list = words

            for index, word in enumerate(words):
                word_aux, emoji_aux = separate_emojis(word)
                if len(emoji_aux) > 0:
                    if len(word_aux) > 0:
                        temp_list[index] = word_aux
                    temp_list = words[:index] + emoji_aux + words[index:]
            corpus.append(temp_list)

    #Getting dataset important variables
    num_of_sentences = len(corpus)
    num_of_words = 0
    for line in corpus:
        num_of_words += len(line)

    print('Num of sentences - %s' % (num_of_sentences))
    print('Num of words - %s' % (num_of_words))

    #Gensim models parameters definition.
    size = 300
    window_size = 5
    epochs = 700
    min_count = 8
    workers = 8

    # train word2vec model using gensim
    model = gensim.models.Word2Vec(corpus, sg=1, window=window_size, size=size,
                     min_count=min_count, workers=workers, iter=epochs, sample=0.01)

    saved_filename = 'word2vec_' + filename + '.p'
    model.save('e2v_finalmodel')

#function to load word2vec model and searches for most similar words
def load_wordvec(filename):
    model = gensim.models.Word2Vec.load('/word_emo2vec/'+ filename)
    while True:
        word = input('Word for similarity, exit() for returning model\n')
        if word == 'exit()':
            break
        else:
            print(model.similar_by_word(word,10))
    return model

#Fuction to create emojis cluster from word2vec model, returns list with lists of each cluster elements
def cluster_wordvec(filename):
    model = load_wordvec(filename)

    #Obtaining emojis vector from model
    X = model[emo_list]
    NUM_CLUSTERS = 6

    #K means cluster creation
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

    print(assigned_clusters)

    #Auxiliary variables, hardcoded must be improved
    clus1 = []
    clus2 = []
    clus3 = []
    clus4 = []
    clus5 = []
    clus6 = []

    #cluster assignment
    for i, word in enumerate(emo_list):
        print(word + ":" + str(assigned_clusters[i]))
        if assigned_clusters[i] == 0:
            clus1.append(word)
        if assigned_clusters[i] == 1:
            clus2.append(word)
        if assigned_clusters[i] == 2:
            clus3.append(word)
        if assigned_clusters[i] == 3:
            clus4.append(word)
        if assigned_clusters[i] == 4:
            clus5.append(word)
        if assigned_clusters[i] == 5:
            clus6.append(word)

    #Displaying clusters in a organized format
    print(clus1,clus2,clus3,clus4,clus5,clus6)
    print('\n')
    print (' '.join(map(str, clus1)))
    print('\n')
    print('\n')
    print(' '.join(map(str, clus2)))
    print('\n')
    print('\n')
    print(' '.join(map(str, clus3)))
    print('\n')
    print('\n')
    print(' '.join(map(str, clus4)))
    print('\n')
    print('\n')
    print(' '.join(map(str, clus5)))
    print('\n')
    print('\n')
    print(' '.join(map(str, clus6)))
    print('\n')

    return [clus1,clus2,clus3,clus4,clus5,clus6]
# Function for returning Emojis/emotion definition variables, we used them several times across the project
def getEmojisCode():
    feliz = [u"\U0001F600", u"\U0001F602", u"\U0001F603", u"\U0001F604", u"\U0001F606", u"\U0001F607", u"\U0001F609",
             u"\U0001F60A", u"\U0001F60B", u"\U0001F60C", u"\U0001F60D", u"\U0001F60E", u"\U0001F60F", u"\U0001F31E",
             u"\u263A", u"\U0001F618", u"\U0001F61C", u"\U0001F61D", u"\U0001F61B", u"\U0001F63A", u"\U0001F638",
             u"\U0001F639", u"\U0001F63B", u"\U0001F63C", u"\u2764", u"\U0001F496", u"\U0001F495", u"\U0001F601",
             u"\u2665"]
    irritado = [u"\U0001F62C", u"\U0001F620", u"\U0001F610", u"\U0001F611", u"\U0001F620", u"\U0001F621",
                u"\U0001F616", u"\U0001F624", u"\U0001F63E"]
    nojo = [u"\U0001F4A9", u"\U0001F92E", u"\U0001F922"]
    medo = [u"\U0001F605", u"\U0001F626", u"\U0001F627", u"\U0001F631", u"\U0001F628", u"\U0001F630", u"\U0001F640"]
    triste = [u"\U0001F614", u"\U0001F615", u"\u2639", u"\U0001F62B", u"\U0001F629", u"\U0001F622", u"\U0001F625",
              u"\U0001F62A", u"\U0001F613", u"\U0001F62D", u"\U0001F63F", u"\U0001F4942"]
    surpresa = [u"\U0001F633", u"\U0001F62F", u"\U0001F635", u"\U0001F632"]

    return feliz, irritado, nojo, medo, triste, surpresa

#Function to create 2D or 3D plot of emoji spacial localization. We perform TSNE reduction to 2 dimensions based on the 300 sized vectors from w2v model
def display_closestwords_tsnescatterplot(model, word, dimension):
    feliz, irritado, nojo, medo, triste, surpresa = getEmojisCode()

    arr = np.empty((0, 300), dtype='f')
    word_labels = word
    close_words = []

    # get close words
    for w in word:
        close_words = close_words + model.similar_by_word(w,topn=1)

    #Looping through close words
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    if dimension == 2:
        # find tsne coords for 2 dimensions
        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(arr)

        x_coords = Y[:, 0]
        y_coords = Y[:, 1]
        # display scatter plot, hardcoded for 6 emotions
        plt.scatter(x_coords, y_coords)
        prop = FontProperties()
        prop.set_file('Symbol.ttf')
        for label, x, y in zip(word_labels, x_coords, y_coords):
            if label in feliz:
                plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=10, color='green')
            else:
                if label in irritado:
                    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=10, color='red')
                else:
                    if label in nojo:
                        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=10, color='grey')
                    else:
                        if label in medo:
                            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=10, color='cyan')
                        else:
                            if label in triste:
                                plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=10, color='blue')
                            else:
                                if label in surpresa:
                                    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=10, color='yellow')
                                else:
                                    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=10, color='black')
        plt.xlim(x_coords.min() - 35, x_coords.max() + 35)
        plt.ylim(y_coords.min() - 35, y_coords.max() + 35)
        lightgreen_patch = mpatches.Patch(color='grey', label='Disgust')
        cyan_patch = mpatches.Patch(color='cyan', label='Fear')
        red_patch = mpatches.Patch(color='red', label='Anger')
        green_patch = mpatches.Patch(color='green', label='Happiness')
        blue_patch = mpatches.Patch(color='blue', label='Sadness')
        yellow_patch = mpatches.Patch(color='yellow', label='Surprise')
        plt.legend(handles=[lightgreen_patch,cyan_patch,red_patch,green_patch,blue_patch,yellow_patch])
        plt.show()
    if dimension == 3:
        # find tsne coords for 3 dimensions
        tsne = TSNE(n_components=3, random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(arr)

        x_coords = Y[:, 0]
        y_coords = Y[:, 1]
        z_coords = Y[:, 2]
        # display scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_coords, y_coords, z_coords)
        for label, x, y, z in zip(word_labels, x_coords, y_coords, z_coords):
            if label in feliz:
                ax.text(x, y, z, label, fontsize=10, color='green')
            else:
                if label in irritado:
                    ax.text(x, y, z, label, fontsize=10, color='red')
                else:
                    if label in nojo:
                        ax.text(x, y, z, label, fontsize=10, color='lightgreen')
                    else:
                        if label in medo:
                            ax.text(x, y, z, label, fontsize=10, color='cyan')
                        else:
                            if label in triste:
                                ax.text(x, y, z, label, fontsize=10, color='blue')
                            else:
                                if label in surpresa:
                                    ax.text(x, y, z, label, fontsize=10, color='yellow')
                                else:
                                    ax.text(x, y, z, label, fontsize=10, color='black')

        lightgreen_patch = mpatches.Patch(color='lightgreen', label='Disgust')
        cyan_patch = mpatches.Patch(color='cyan', label='Fear')
        red_patch = mpatches.Patch(color='red', label='Anger')
        green_patch = mpatches.Patch(color='green', label='Happiness')
        blue_patch = mpatches.Patch(color='blue', label='Sadness')
        yellow_patch = mpatches.Patch(color='yellow', label='Surprise')
        ax.legend(handles=[lightgreen_patch, cyan_patch, red_patch, green_patch, blue_patch, yellow_patch])
        plt.show()

#Auxiliary function to get emojis using regular expressions, accepts string of a text and returns a string without emojis and a list of existing emojis in the input text
def get_emojis(text):
    existing_emo = []
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U0001F1F2-\U0001F1F4"  # Macau flag
                               u"\U0001F1E6-\U0001F1FF"  # flags
                               u"\U0001F600-\U0001F64F"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U0001F1F2"
                               u"\U0001F1F4"
                               u"\U0001F620"
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\U0001F910-\U0001F924"
                               "]+", flags=re.UNICODE)


    splitstring = text.split()
    for item in splitstring:
        #print(item)
        if item in emo_list:
            existing_emo.append(item)
            splitstring.remove(item)
    retstring = " ".join(splitstring)
    retstring= emoji_pattern.sub(r'', retstring)
    return retstring, existing_emo

#Auxiliary function for dealing with cases where we have two joined emojis, accepts string of a given text, returns string of the cleaned text without
#emojis and list of existing emojis
def separate_emojis(word):
    sent_emoji = []
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U0001F1F2-\U0001F1F4"  # Macau flag
                               u"\U0001F1E6-\U0001F1FF"  # flags
                               u"\U0001F600-\U0001F64F"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U0001F1F2"
                               u"\U0001F1F4"
                               u"\U0001F620"
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\U0001F910-\U0001F924"
                               "]+", flags=re.UNICODE)
    ret_word = emoji_pattern.sub(r'', word)
    emojis = emoji_pattern.findall(word)
    for emo in emojis:
        sent_emoji = sent_emoji + list(emo)
    else:
        return ret_word, sent_emoji

#Auxiliary function for dealing with joined emojis
def sep(word):
    print(word)
    a = list(word)
    print(word.encode('unicode-escape').decode('ASCII'))
    return a

#Auxiliary function that removes @RT twitter tokens. Accepts a string of a given text/tweet and returns the same text without RT tokens
def remove_rt(text):
    remrt = text.replace('RT @USERNAME', '')
    remrt = " ".join(remrt.split())
    return remrt

"""
Function to create a final processed dataset. Accepts two string parameters of both input and output files names. Will use great deal of fucntions to
extract additional features from the texts and removing inconsistencies. Will take a while to process.
"""
def readDataset(inputfilename, outputfilename):
    with open(inputfilename, 'r', encoding='utf8') as f, open(outputfilename, 'w', encoding='utf8') as out_file:
        #variables to file read and write
        reader = csv.reader(f)
        writer = csv.writer(out_file)
        writer.writerow(['Username', 'Data', 'Tweet Original', 'Tweet Limpo', 'Emocao', 'Localizacao', 'Emoji', 'NRCValence'
                         , 'NRCArousal', 'NRCDominance', 'ANEWValence'
                         , 'ANEWArousal', 'ANEWDominance', 'Sentiment' ,'Negacoes', 'Total Palavras', 'Maior Palavra', 'Media de Carateres'
                         , 'Pontos de Exclamacao', 'Palavras Maiusculas', 'Posto de Interrogacao'])
        #tqdm helps displaying the current state of the data processing time
        for row in tqdm(reader):
            #Data inconsistencies are ignored
            if len(row) < 1:
                continue
            #Additional features and data cleanse processes
            cleanedtext = remove_rt(row[2])
            cleanedtext, emojilist = get_emojis(cleanedtext)
            val_nrc, aro_nrc, dom_nrc, val_anew, aro_anew, dom_anew, sentiment = analyzefile(row[2])
            cleanedtext = transform_elongated_word(cleanedtext)
            negcount = count_neg_words(cleanedtext)
            totalwords, maxchar, avgchar = count_word_length(cleanedtext)
            exclacount = count_exclamation_letter(cleanedtext)
            uppercount = count_uppercase_words(row[2])
            intercount = count_interrogation_letter(cleanedtext)
            #File write
            if len(emojilist) == 0:
                writer.writerow([row[0], row[1], row[2], cleanedtext ,row[3], row[4], 'None', val_nrc, aro_nrc, dom_nrc, val_anew, aro_anew, dom_anew, sentiment, negcount, totalwords, maxchar, avgchar, exclacount, uppercount, intercount])
            else:
                writer.writerow(
                    [row[0], row[1], row[2], cleanedtext, row[3], row[4], emojilist[0], val_nrc, aro_nrc, dom_nrc, val_anew, aro_anew, dom_anew, sentiment, negcount, totalwords,
                     maxchar, avgchar, exclacount, uppercount, intercount])


#Function to display dataset class distributions, helps to understand our datasets, creates piechart with class distribution
def dataStat(filename,label):
    all_data = read_data(filename,False)
    datashape = all_data[label].value_counts()
    print(datashape.shape)


    labels = ['Happiness', 'Disgust', 'Anger', 'Fear', 'Sadness', 'Surprise']
    sizes = list(datashape)
    colors = ['seagreen', 'chartreuse', 'red','blueviolet', 'deepskyblue', 'gold']
    explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    plt.pie(sizes, colors=colors, labels=labels, labeldistance=1.10 ,autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode=explode)
    # draw circle
    plt.tight_layout()
    plt.show()

#Function to select a lower ammount of emojis for analysing the performance of the developed models
def selectData(filename,low_emo):
    feliz, irritado, nojo, medo, triste, surpresa = getEmojisCode()

    all_data = read_data(filename, False)

    print(all_data['Total Palavras'])
    print(all_data['Total Palavras'].mean())

    #Replacing joy class cat emojis
    all_data['Emoji'] = all_data['Emoji'].replace([u"\U0001F638",
             u"\U0001F639", u"\U0001F63B", u"\U0001F63C",u"\U0001F63A"],[u"\U0001F600",u"\U0001F602",u"\U0001F60D",u"\U0001F609",u"\U0001F603"])

    #Replacing remaining cat emojis
    all_data['Emoji'] = all_data['Emoji'].replace([u"\U0001F63F",u"\U0001F63E",u"\U0001F640"],[u"\U0001F625",u"\U0001F620",u"\U0001F631"])

    all_data = all_data[all_data['Emoji'] != 'None']

    a = all_data['Emoji'].value_counts()
    for i in range(len(a)):
        if a.index[i] in feliz:
            feliz = a.index[i]
        if a.index[i] in irritado:
            irritado = a.index[i]
        if a.index[i] in nojo:
            nojo = a.index[i]
        if a.index[i] in medo:
            medo = a.index[i]
        if a.index[i] in triste:
            triste = a.index[i]
        if a.index[i] in surpresa:
            surpresa = a.index[i]

    if low_emo:
        small_emo = feliz + irritado + nojo + medo + triste + surpresa
        print(list(small_emo))
        all_data = all_data.loc[all_data['Emoji'].isin(list(small_emo))]
        print(all_data)

        print(all_data['Emoji'])
        print(all_data['Emoji'].value_counts())

    return all_data

#Fuction to load news dataset for additional experiments on our models
def newDataset():
    header = ['Tweet Limpo', 'Emocao']

    all_data = pd.read_csv('Base_2000.csv', names=header, delimiter=';')
    all_data = all_data.replace(['Alegria','Tristeza','Raiva','Desgosto','Surpresa','Medo'], ['feliz','triste','irritado','nojo','surpresa','medo'])
    all_data = all_data[all_data['Emocao'] != 'Neutro']
    return all_data

#----------------------------------------------------------------------------------
#Main function to call developed functions, adapt to your own necesseties.

if __name__ == '__main__':
    pass
    #readDataset()
    #create_model('NB',1,2, 'Emocao',False,False)
    #load_wordvec()
    #create_wordvec()
    #showSparseMatrix(1,3,0)
    #findSimilarSentence()
    #create_lstm('Emocao',200,True)
    #dataStat()
    #balanceData('Emocao')
    #create_balanced_model('NB',1,2,'Emocao',False)
    #selectData(True)
    #cluster_wordvec()
    #newDataset()

