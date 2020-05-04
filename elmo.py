import string
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import scipy
import model_definitions
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from allennlp.commands.elmo import ElmoEmbedder

"""
Script consisting of additional experiments using Elmo and Bert word embeddings and retraining new models for emotion
classification. Still in implementation phase.
"""

#TFhub elmo definition
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

#Function to remove punction from input string.
def remove_punc(text):
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    text = ''.join([i for i in text if not i.isdigit()])
    return text

#Transforms a given string composing of a text to ELMO word embeddings. Returns vector of embeddings
def elmo_vectors(x):
  x = x['Text'].tolist()
  embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    # return average of ELMo features
    return sess.run(tf.reduce_mean(embeddings,1))

#Auxiliary function to convert dataframe to list of list format. Pandas method was not working properly
def convert_rows(dataframe):
    return_list = []

    # Iterate over each row
    for index, rows in dataframe.iterrows():
        # Create list for the current row
        my_list = str(rows.Text).split()

        # append the list to the final list
        return_list.append(my_list)
    return return_list

#Function for reading dataset and trasform all texts to its Elmo embedding vectors. We train a Logistic Regression model for simple performance analysis
def process_data(filename, save):
    all_data = model_definitions.read_data(filename)
    all_data['Text'] = all_data['Text'].values.astype('U')
    all_data['Text'] = all_data['Text'].apply(remove_punc)

    #Train and test sets split

    train, test = train_test_split(all_data, test_size=0.20, shuffle=True)

    #Creating batches of 100 due to performance issues
    list_train = [train[i:i + 100] for i in range(0, train.shape[0], 100)]
    list_test = [test[i:i + 100] for i in range(0, test.shape[0], 100)]

    print("Converting text to Elmo")

    elmo_train = [elmo_vectors(x) for x in tqdm(list_train)]
    elmo_test = [elmo_vectors(x) for x in tqdm(list_test)]

    #Saving embeddings
    if save:
        print("Saving Elmo embeddigns")

        elmo_train_new = np.concatenate(elmo_train, axis=0)
        elmo_test_new = np.concatenate(elmo_test, axis=0)

        # save elmo_train_new
        pickle_out = open("elmo_train.pickle", "wb")
        pickle.dump(elmo_train_new, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()

        # save elmo_test_new
        pickle_out = open("elmo_test.pickle", "wb")
        pickle.dump(elmo_test_new, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()

    print("Training Model")

    #Logistic Regression model creation and evaluation
    lreg = LogisticRegression()
    lreg.fit(elmo_train_new, train['Emocao'])

    preds_valid = lreg.predict(elmo_test_new)
    print(f1_score(test['Emocao'], preds_valid))


#Test function converting Texts to Elmo embeddings using AllenNLP
def use_allen():
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/elmo_pt_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/elmo_pt_weights.hdf5"


    eltest = ElmoEmbedder(options_file, weight_file)

    test_list = ['First test']
    sec_test = ['Second test']

    vec = eltest.embed_sentence(test_list)
    vec2 = eltest.embed_sentence(sec_test)

    print(vec)
    print(vec2)
    print(scipy.spatial.distance.cosine(vec,vec2))


def read_embeddings():
    infile = open('')


if __name__ == '__main__':
    pass
    #read_data()
    #use_allen()
    #statistics()