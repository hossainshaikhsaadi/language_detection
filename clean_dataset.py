import pandas as pd 
import re

import warnings
warnings.filterwarnings("ignore")

def get_data():
    # read the dataframe and returns the text and language column as lists
    df = pd.read_csv("Language Detection.csv")
    return df['Text'].tolist(), df['Language'].tolist()

def clean_text(texts):
    # cleans the text
    for i in range(len(texts)):
        single_text = texts[i] 
        single_text = re.sub('[0-9]+', '', single_text)
        single_text = re.sub('[()]', '', single_text)
        single_text = re.sub('[[]]', '', single_text)
        single_text = re.sub('[^\w\s]', ' ', single_text)
        single_text = single_text.strip()
        single_text = single_text.lower()
        single_text = single_text.replace('\n','')
        texts[i] = single_text
    return texts

def make_vocabulary(texts):
    # makes vocabulary from the texts
    vocab = []
    vocab.append("<pad>")
    vocab.append("<unk>")
    for text in texts:
        text = text.strip()
        t_l = text.split()
        n_t = [t for t in t_l if t not in vocab]
        vocab.extend(n_t)
    return vocab

def dump_texts(texts):
    # writes the texts in a file
    text_file = open('data', 'w')
    for text in texts:
        text_file.write(text+ '\n')

def dump_labels(languages):
    # writes the labels in a file
    language_file = open('label', 'w')
    for language in languages:
        language_file.write(language+ '\n')

def dump_vocab(vocab):
    # writes the vocabulary in a file
    vocab_file = open('vocab', 'w')
    for word in vocab:
        vocab_file.write(word+ '\n')
    

def get_data_labels_vocab():
    #returns all the texts, labels and a vocabulary
    texts, languages = get_data()
    texts = clean_text(texts)
    vocab = make_vocabulary(texts)
    return texts, languages, vocab

    

if __name__ == "__main__":

    # get all the texts, labels and vocabulary
    texts, languages, vocab = get_data_labels_vocab()
    # write the texts, labels and vocabulary in files
    dump_texts(texts)
    dump_vocab(vocab)
    dump_labels(languages)





