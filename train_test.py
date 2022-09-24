import torch
import torch.nn as nn
from torch import optim
from get_dataloader import get_dataloaders
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score
from pytorch_models import BiLSTM, Bert,Dbert 
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel

# setting the seeds for torch, random and numpy for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def get_device():
    # returns the device cuda or cpu
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device

def train_bilstm(model, optimizer, dataloader, epochs, model_name, validation_dataloader):
    # function to train a the BiLSTM model
    device = get_device()
    min_loss = 10000
    for epoch in range(epochs):
        running_loss = 0.0
        total_instances = 0
        model.train()
        for i, data in enumerate(dataloader):
            _ , word_indexes, labels = data
            total_instances = total_instances + len(labels)

            word_indexes = torch.tensor(word_indexes)
            labels = torch.tensor(labels)

            #takes the data and labels to cuda if available
            word_indexes = word_indexes.to(device)
            labels = labels.to(device)

            pred = model(word_indexes)
            loss = F.cross_entropy(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = running_loss + loss.item()

            if i % 10 == 0:
                sentence = "BiLSTM : Training - Epoch: "+str(epoch+1)+", Iteration : "+str(i+1)+" Loss : "+str(loss.item())
                sentence = sentence + " Running Loss : "+str(running_loss)
                print(sentence)

            #if i >= 10:
            #    break
        print("Validation ongoing for Epoch : "+str(epoch+1))
        # validate after each epoch
        validation_loss = validate_bilstm(epoch, model, validation_dataloader)
        print("Validation done for Epoch : "+str(epoch+1))
        if validation_loss < min_loss:
            # save the based on validation loss model with least validation loss is saved
            min_loss = validation_loss
            save_model(model, model_name)

def train_plm(model, tokenizer, optimizer, dataloader, epochs, model_name, model_save_name, validation_dataloader):
    # function to train a the BERT or DistilBERT based model
    device = get_device()
    min_loss = 10000
    for epoch in range(epochs):
        running_loss = 0.0
        total_instances = 0
        model.train()
        for i, data in enumerate(dataloader):
            text , _ , labels = data
            total_instances = total_instances + len(labels)

            encoded_sentences = tokenizer(text, return_tensors = "pt", padding=True, truncation=True)
            labels = torch.tensor(labels)
            encoded_sentences = encoded_sentences.to(device)
            labels = labels.to(device)

            pred = model(encoded_sentences)
            loss = F.cross_entropy(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss = running_loss + loss.item()

            if i % 10 == 0:
                if "distil" in model_name:
                    sentence = "DistilBERT : Training - Epoch: "+str(epoch+1)+", Iteration : "+str(i+1)+" Loss : "+str(loss.item())
                else:
                    sentence = "BERT : Training - Epoch: "+str(epoch+1)+", Iteration : "+str(i+1)+" Loss : "+str(loss.item())
                sentence = sentence + " Running Loss : "+str(running_loss)
                print(sentence)

            #if i >= 10:
            #    break
        print("Validation ongoing for Epoch : "+str(epoch+1))
        # validate after each epoch
        validation_loss = validate_plm_model(epoch, model, tokenizer, validation_dataloader)
        print("Validation done for Epoch : "+str(epoch+1))
        if validation_loss < min_loss:
            # save the based on validation loss model with least validation loss is saved
            min_loss = validation_loss
            save_model(model, model_save_name)

def validate_bilstm(epoch, model, dataloader):
    # validate the BiLSTM model
    device = get_device()
    running_loss = 0.0
    total_instances = 0
    model.eval()
    for i, data in enumerate(dataloader):
        _ , word_indexes, labels = data
        total_instances = total_instances + len(labels)

        word_indexes = torch.tensor(word_indexes)
        labels = torch.tensor(labels)
        word_indexes = word_indexes.to(device)
        labels = labels.to(device)

        pred = model(word_indexes)
        loss = F.cross_entropy(pred, labels)

        running_loss = running_loss + loss.item()

    validation_loss = running_loss/total_instances

    return validation_loss

def validate_plm_model(epoch, model, tokenizer, dataloader):
    # validate the BERT or DistilBERT based model
    device = get_device()
    running_loss = 0.0
    total_instances = 0
    model.eval()
    for i, data in enumerate(dataloader):
        text , _ , labels = data
        total_instances = total_instances + len(labels)

        encoded_sentences = tokenizer(text, return_tensors = "pt", padding=True, truncation=True)
        labels = torch.tensor(labels)

        encoded_sentences = encoded_sentences.to(device)
        labels = labels.to(device)

        pred = model(encoded_sentences)
        loss = F.cross_entropy(pred, labels)

        running_loss = running_loss + loss.item()

    validation_loss = running_loss/total_instances

    return validation_loss

def test_bilstm(model, dataloader):
    # test the BiLSTM model
    device = get_device()
    true_labels = []
    predicted_labels = []
    print("Testing the model")
    model.eval()
    for i, data in enumerate(dataloader):
        _ , word_indexes, labels = data
        word_indexes = torch.tensor(word_indexes)
        word_indexes = word_indexes.to(device)
        labels = torch.tensor(labels)
    
        logits = model(word_indexes)
        predictions = torch.argmax(logits, dim=1)
        predictions = predictions.tolist()

        predicted_labels.extend(predictions)
        true_labels.extend(labels)
    print("Testing Complete")
    return true_labels, predicted_labels

def test_plm_model(model, tokenizer, dataloader):
    # test the BERT or DistilBERT based model
    device = get_device()
    true_labels = []
    predicted_labels = []
    print("Testing the model")
    model.eval()
    for i, data in enumerate(dataloader):
        text , _ , labels = data

        encoded_sentences = tokenizer(text, return_tensors = "pt", padding=True, truncation=True)
        labels = torch.tensor(labels)
        
        encoded_sentences = encoded_sentences.to(device)

        logits = model(encoded_sentences)
        predictions = torch.argmax(logits, dim=1)
        predictions = predictions.tolist()

        predicted_labels.extend(predictions)
        true_labels.extend(labels)
    print("Testing Complete")
    return true_labels, predicted_labels

def evaluation_scores(labels, predictions, class_names):
    # evaluates models predictions using true labels, returns a classification report and micro F1 and macro F1 score
    report = classification_report(labels, predictions, target_names = class_names)
    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    return report, micro_f1, macro_f1

def save_model(model, model_name):
    # saves a model
    torch.save(model.state_dict(), model_name)

def load_bilstm_model(vocab_length, model_name):
    # load the BiLSTM model
    device = get_device()
    embedding_dimension = 300
    vocabulary_size = vocab_length
    hidden_dimension = 128
    num_classes = 17    
    model = BiLSTM(embedding_dimension, vocab_length, hidden_dimension, num_classes)
    model.to(device)
    model.load_state_dict(torch.load(model_name))
    model.to(device)
    return model

def load_plm_model(model_name):
    # load the BERT or DistilBERT based model
    device = get_device()
    if model_name == "bert.pth":
        model = Bert(num_classes = 17)
    else:
        model = Dbert(num_classes = 17)
    model.to(device)
    model.load_state_dict(torch.load(model_name))
    model.to(device)
    return model

def bilstm_model_optimizer(vocab_length):

    # creates bilstm model and optimizer
    device = get_device()
    embedding_dimension = 300
    vocabulary_size = vocab_length
    hidden_dimension = 128
    num_classes = 17

    bilstm_model = BiLSTM(embedding_dimension, vocab_length, hidden_dimension, num_classes)
    bilstm_model.to(device)
    parameters = list(bilstm_model.parameters())
    optimizer = optim.Adam(parameters, lr = 0.001)
    return bilstm_model, optimizer

def plm_model_optimizer(model_name):
    # creates BERT or DistilBERT based model and optimizer
    device = get_device()
    if "distil" in model_name:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = Dbert(num_classes=17)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = Bert(num_classes=17)
    model.to(device)
    parameters = list(model.parameters())
    optimizer = optim.Adam(parameters, lr = 0.0001)

    return model, tokenizer, optimizer

def bilstm_pipeline(epochs, batch_size):

    # whole pipeline to train, validate and test a BiLSTM model
    model_name = "bilstm_model.pth"
    train_dataloader, validation_dataloader, test_dataloader, vocab_length, class_names = get_dataloaders(batch_size)

    model, optimizer = bilstm_model_optimizer(vocab_length)
    print("Training BiLSTM model")
    train_bilstm(model, optimizer, train_dataloader, epochs, model_name, validation_dataloader)
    print("Training BiLSTM model Complete")

    model = load_bilstm_model(vocab_length, model_name)
    print("Testing BiLSTM model")
    labels, predictions = test_bilstm(model, test_dataloader)
    print("Testing BiLSTM model complete")
    print("Evaluating BiLSTM model")
    report, micro_f1, macro_f1 = evaluation_scores(labels, predictions, class_names)
    print("Evaluating BiLSTM model complete")
    print("BiLSTM Classification Score Report : ")
    print(report)
    print("\n BiLSTM F1 Micro Score :  "+str(micro_f1))
    print("\n BiLSTM F1 Macro Score :  "+str(macro_f1))
    print("\n\n")


def plms_pipelines(epochs, batch_size):
    # whole pipeline to train, validate and test a BERT or DistilBERT based model
    model_names = ["distilbert-base-multilingual-cased","bert-base-multilingual-cased"]
    model_save_names = ["dbert.pth", "bert.pth"]
    for model_name, model_save_name in zip(model_names, model_save_names):
        train_dataloader, validation_dataloader, test_dataloader, vocab_length, class_names = get_dataloaders(batch_size)
        model, tokenizer, optimizer = plm_model_optimizer(model_name)
        print("Training "+str(model_name.split("-")[0])+" model")
        train_plm(model, tokenizer, optimizer, train_dataloader, epochs, model_name, model_save_name, validation_dataloader)
        print("Training "+str(model_name.split("-")[0])+" model Complete")
        model = load_plm_model(model_save_name)
        print("Testing "+str(model_name.split("-")[0])+" model")
        labels, predictions = test_plm_model(model, tokenizer, test_dataloader)
        print("Testing "+str(model_name.split("-")[0])+" model Complete")
        print("Evaluating "+str(model_name.split("-")[0])+" model")
        report, micro_f1, macro_f1 = evaluation_scores(labels, predictions, class_names)
        print("Evaluating "+str(model_name.split("-")[0])+" model Complete")
        if "distil" in model_name:
            print("DistilBERT Classification Score Report : ")
            print(report)
            print("\n DistilBERT F1 Micro Score :  "+str(micro_f1))
            print("\n DistilBERT F1 Macro Score :  "+str(macro_f1))
            print("\n\n")
        else:
            print("BERT Classification Score Report : ")
            print(report)
            print("\n BERT F1 Micro Score :  "+str(micro_f1))
            print("\n BERT F1 Macro Score :  "+str(macro_f1))
            print("\n\n")

if __name__ == "__main__":
    # whole training, validation and testing pipeline for BiLSTM model
    bilstm_pipeline()
    # whole training, validation and testing pipeline for BERT and DistilBERT based models
    plms_pipelines()
    
    

        










        