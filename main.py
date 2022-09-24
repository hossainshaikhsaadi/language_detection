from sklearn_models import train_all_sklearn_models
from train_test import bilstm_pipeline, plms_pipelines

# train and test all the sklearn models (Logistic Regression, Random Forest, Logistic Regression, Support Vector Machine
# and voting classifier consists of Logistic Regression, Random Forest, Logistic Regression, Support Vector Machine
# with soft voting
train_all_sklearn_models()

# train, validate and test the BiLSTM model
bilstm_pipeline(epochs = 10, batch_size = 32)

# train, validate and test the BERT based model and the distilBERT based model
plms_pipelines(epochs = 2, batch_size = 8)