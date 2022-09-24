from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from clean_dataset import clean_text, get_data
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

def clean_data():
    # clean the data (text and language)
    texts, languages = get_data()
    texts = clean_text(texts)
    return texts, languages

def vectorizer_encoder():
    # creates object of CountVectorizer and LabelEncoder 
    vectorizer = CountVectorizer()
    label_encoder = LabelEncoder()
    return vectorizer, label_encoder

def get_data_label(vectorizer, label_encoder, texts, languages):
    # creates data matrix using count vectorizer object and labels using label encoder object
    data_matrix = vectorizer.fit_transform(texts)
    labels = label_encoder.fit_transform(languages)
    return data_matrix, labels

def get_train_test_data(data_matrix, labels):
    # splits the data into test and train set 
    X_train, X_test, y_train, y_test = train_test_split(data_matrix, labels, test_size = 0.20, random_state = 42)
    return X_train, X_test, y_train, y_test

def evaluation_scores(y_test, y_pred, label_encoder):
    # evaluates models predictions using true labels, returns a classification report and micro F1 and macro F1 score
    labels = label_encoder.inverse_transform([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    report = classification_report(y_test, y_pred, target_names = labels)
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    return report, micro_f1, macro_f1

def train_model(model, data, labels):
    # train a model using the given data and labels 
    model.fit(data, labels)
    return model    

def test_model(model, data):
    # predicts the labels of the given data
    y_pred = model.predict(data)
    return y_pred

def pipeline(model, train_data, train_labels, test_data, test_labels, model_name, label_encoder):
    # whole pipeline to train and test a model
    model = train_model(model, train_data, train_labels)
    y_pred = test_model(model, test_data)
    report, micro_f1, macro_f1 = evaluation_scores(test_labels, y_pred, label_encoder)
    print(str(model_name)+" Score Report : ")
    print(report)
    print("\n" + str(model_name)+" F1 Micro Score :  "+str(micro_f1))
    print("\n" + str(model_name)+"  F1 Macro Score :  "+str(macro_f1))
    print("\n\n")

def get_experiment_data():
    # get all the data and returns train and test set
    texts, languages = clean_data()
    vectorizer, label_encoder = vectorizer_encoder()
    data_matrix, labels = get_data_label(vectorizer, label_encoder, texts, languages)
    X_train, X_test, y_train, y_test = get_train_test_data(data_matrix, labels)
    return X_train, X_test, y_train, y_test, label_encoder

def train_all_sklearn_models():
    #get train and test data
    X_train, X_test, y_train, y_test, label_encoder = get_experiment_data()

    #Multinomial Naive Bayes
    mnb_model = MultinomialNB()
    pipeline(mnb_model,X_train, y_train, X_test, y_test, "Multinomial Naive Bayes", label_encoder)
    
    #Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    pipeline(rf_model,X_train, y_train, X_test, y_test, "Random Forest", label_encoder)


    #Logistic Regression
    lr_model = LogisticRegression()
    pipeline(lr_model,X_train, y_train, X_test, y_test, "Logistic Regression", label_encoder)

    #Support Vector Machine
    svc_model = SVC(kernel = "linear")
    pipeline(svc_model,X_train, y_train, X_test, y_test, "Support Vector Machine", label_encoder)
    
    #voting classifier
    # the voting classifier is consists of Multinomial Naive Bayes, Logistic Regression, Support Vector Machine 
    # and Random Forest classifier and uses soft voting 
    svc_model = SVC(kernel = "linear", probability=True)
    lr_model = LogisticRegression()
    rf_model = RandomForestClassifier(random_state=42)
    mnb_model = MultinomialNB()
    voting_model = VotingClassifier(estimators=[('mnb', mnb_model), ('rf', rf_model), ('lr', lr_model), ('svc', svc_model)], voting='soft')
    pipeline(voting_model,X_train, y_train, X_test, y_test, "Voting Classifier", label_encoder)


if __name__ == "__main__":

    # train, test and evaluate all the sklearn models
    train_all_sklearn_models()

