from torch.utils.data import Dataset, DataLoader
from clean_dataset import get_data_labels_vocab
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import random
import numpy as np

# setting the seeds for torch, random and numpy for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


class LanguageDetectionDataset(Dataset):
    # class for the dataloader
    def __init__(self, text_data, labels, vocab):
        self.text_data = text_data
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = self.text_data[idx].strip()
        label = self.labels[idx]
        words = text.split()
        text_length = len(words)
        word_indexes = [self.vocab[word] for word in words]
        return text, word_indexes, label, text_length

def process_batch(batch):
    # collate function for processing a batch 
    text, word_indexes, label, text_length = zip(*batch)
    text_length = list(text_length)
    word_indexes = list(word_indexes)
    text = list(text)
    max_len = max(text_length)
    label = list(label)
    for i in range(len(word_indexes)):
        single_sentence = word_indexes[i]
        sentence_length = max_len - len(single_sentence)
        if sentence_length > 0:
            zeros = [0]*sentence_length
            word_indexes[i].extend(zeros)
    return text, word_indexes, label

def make_dictionary(vocabulary):
    # function to make a word to index dictionary
    word_to_index = dict()
    for i in range(len(vocabulary)):
        word_to_index[vocabulary[i]] = i
    return word_to_index

def get_experiment_data():
    # return all the data required for creating a dataloader
    texts, languages, vocab = get_data_labels_vocab()
    vocab_length = len(vocab)
    word_to_index = make_dictionary(vocab)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(languages)
    class_names = label_encoder.inverse_transform([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    return texts, labels, word_to_index, class_names, vocab_length

def get_dataloaders(batch_size):
    # returns the train, validation and test dataloader
    texts, labels, vocab_dict, class_names, vocab_length = get_experiment_data()

    X, X_test, Y, y_test = train_test_split(texts, labels, test_size = 0.20, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = 0.125, random_state = 42)
    
    language_detection_dataset_train = LanguageDetectionDataset(X_train, y_train, vocab_dict)
    train_dataloader = DataLoader(language_detection_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn = process_batch)
    
    language_detection_dataset_test = LanguageDetectionDataset(X_test, y_test, vocab_dict)
    test_dataloader = DataLoader(language_detection_dataset_test, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn = process_batch)

    language_detection_dataset_val = LanguageDetectionDataset(X_val, y_val, vocab_dict)
    validation_dataloader = DataLoader(language_detection_dataset_val, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn = process_batch)

    return train_dataloader, validation_dataloader, test_dataloader, vocab_length, class_names
    

