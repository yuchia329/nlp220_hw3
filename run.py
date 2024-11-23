import torch
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from io import StringIO
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import Dataset
from utils import setSeed, log
from loaddata import prepareData, makeDateloader

# Initialize a StringIO buffer to collect data
stats_buffer = StringIO()

class TextDataset(Dataset):
    def __init__(self, texts, labels, vectorizer, fit_vectorizer=False):
        self.texts = texts
        self.labels = labels
        self.vectorizer = vectorizer
        # Only fit the vectorizer on the training data        
        if fit_vectorizer:
            self.features = self.vectorizer.fit_transform(self.texts).toarray()
        else:
            self.features = self.vectorizer.transform(self.texts).toarray()
    
    def x_training_vec(self):
        return torch.tensor(self.features, dtype=torch.float32)
    
    def y_training_vec(self):
        return torch.tensor(self.labels.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

def training(model, train_loader, vali_loader):
    train_features = []
    train_labels = []

    for features, labels in train_loader:
        train_features.extend(features.numpy())
        train_labels.extend(labels.numpy())
        
def evaluation(model, x_test_vec, y_test_vec):
    model.eval()
    with torch.no_grad():
        outputs = model(x_test_vec)
        predicted = (outputs > 0.0).int()
        p, r, f1, _ = precision_recall_fscore_support(y_true=y_test_vec, y_pred=predicted, average='macro')
        print("precision: ", p)
        print("recall: ", r)
        print("f1: ", f1)

def selectModels(feature_dim, label_dim):
    models = [
        # (MultiOutputClassifier(MultinomialNB(alpha=0.100009816904796)), MultinomialNB.__name__),
        # (MultiOutputClassifier(SGDClassifier(max_iter=3000, tol=1e-3, random_state=42, n_jobs=-1)), SGDClassifier.__name__),
        # (MultiOutputClassifier(DecisionTreeClassifier(random_state=42)), DecisionTreeClassifier.__name__),
        (MultiOutputClassifier(LogisticRegression(C=1, penalty='l2', solver='lbfgs', random_state=42)), LogisticRegression.__name__),
        # (MultiOutputClassifier(ExtraTreesClassifier(n_estimators=85, criterion='log_loss', max_depth=142, max_features='sqrt', random_state=42)), ExtraTreesClassifier.__name__),
    ]
    return models

def eval(model, x_vec, y_vec, reduce_labels_rate, test=False):
    dataset = "Test" if test else "Validation"
    stats_buffer.write(f'{dataset} Set\n')
    predict_begin = time.time()
    log('predict_begin: ', predict_begin)
    y_pred = model.predict(x_vec)
    predict_end = time.time()
    predict_duration = predict_end - predict_begin
    log('predict_end: ', predict_end, ' duration: ', predict_duration)
    stats_buffer.write(f'Predict Duration: {predict_duration:3f}\n')
    
    # pad 0 on y_pred if reduce_labels_rate > 0
    if reduce_labels_rate > 0:
        pred_label_size = y_pred.shape[1]
        true_label_size = y_vec.shape[1]
        y_pred = np.pad(y_pred, ((0, 0), (0, true_label_size-pred_label_size)), mode='constant', constant_values=0)
    stats_buffer.write(f'Reduce Labels Rate: {reduce_labels_rate}\n')
        
    # Evaluate on the data
    micro_f1 = f1_score(y_vec, y_pred, average='micro')
    macro_f1 = f1_score(y_vec, y_pred, average='macro')
    report = classification_report(y_vec, y_pred, zero_division=0)
    log(" f1 micro: ", micro_f1, " f1 macro: ", macro_f1)#, " Classification Report: ", report)
    stats_buffer.write(f'f1 micro: {micro_f1}\nf1 macro: {macro_f1}\nClassification Report:\n{report}\n')
    
    # p, r, f1, _ = precision_recall_fscore_support(y_true=y_val_vec, y_pred=y_pred)
    # log("precision: ", p, " recall: ", r, " f1_score: ", f1)
    # p, r, f1, _ = precision_recall_fscore_support(y_true=y_val_vec, y_pred=y_pred, average="macro")
    # log("macro precision: ", p, " recall: ", r, " f1_score: ", f1)
    # p, r, f1, _ = precision_recall_fscore_support(y_true=y_val_vec, y_pred=y_pred, average="micro")
    # log("micro precision: ", p, " recall: ", r, " f1_score: ", f1)

def pytorchModel(model, x_train_vec, x_val_vec, x_test_vec, y_train_vec, y_val_vec, y_test_vec, reduce_labels_rate):
    XY_Sets = [
        (x_train_vec, y_train_vec, True),
        (x_val_vec, y_val_vec, False),
        (x_test_vec, y_test_vec, False),
        ]
    [[train_loader, val_loader, test_loader], [train_dataset, val_dataset, test_dataset]] = makeDateloader(XY_Sets)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # train loop
    num_epoch = 150
    fit_begin = time.time()
    log('fit_begin: ', fit_begin)
    for epoch in range(num_epoch):

        # mini-batch gradient decent
        running_loss = torch.tensor(0.)
        for x_batch, y_batch in train_loader:

            # forward pass
            model.train()
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            running_loss += loss

            # backward pass
            loss.backward()
            optimizer.step()

        # evalute in mini batches
        model.eval()
        with torch.no_grad():
            total_correct = 0
            for x_batch, y_batch in test_loader:
                outputs = model(x_batch)
                predicted = (outputs > 0.0).int()
                total_correct += (predicted == y_batch).sum()

        log(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Acc: {total_correct / len(test_dataset)}')
    fit_end = time.time()
    fit_duration = fit_end - fit_begin
    log('fit_end: ', fit_end, ' duration: ', fit_duration)
    stats_buffer.write(f'Fit Duration: {fit_duration:3f}\n')
    model.eval()
    with torch.no_grad():
        predict_begin = time.time()
        log('predict_begin: ', predict_begin)
        outputs = model(x_test_vec)
        predicted = (outputs > 0.0).int()
        predict_end = time.time()
        predict_duration = predict_end - predict_begin
        log('predict_end: ', predict_end, ' duration: ', predict_duration)
        stats_buffer.write(f'Predict Duration: {predict_duration:3f}\n')
        # Evaluate on the data
        micro_f1 = f1_score(y_test_vec, predicted, average='micro')
        macro_f1 = f1_score(y_test_vec, predicted, average='macro')
        report = classification_report(y_test_vec, predicted, zero_division=0)
        log(" f1 micro: ", micro_f1, " f1 macro: ", macro_f1)#, " Classification Report: ", report)
        stats_buffer.write(f'f1 micro: {micro_f1}\nf1 macro: {macro_f1}\nClassification Report:\n{report}\n')
        
def trainAndEval(x_train_vec, x_val_vec, x_test_vec, y_train_vec, y_val_vec, y_test_vec, reduce_labels_rate=0):
    models = selectModels(x_train_vec.shape[1], y_train_vec.shape[1])
    for (model, name) in models:
        if name =="MLP":
            log("Model: ", name)
            stats_buffer.write(f'Model: {name}\n')
            pytorchModel(model, x_train_vec, x_val_vec, x_test_vec, y_train_vec, y_val_vec, y_test_vec, reduce_labels_rate)
        else:
            log("Model: ", name)
            stats_buffer.write(f'Model: {name}\n')
            # train
            fit_begin = time.time()
            log('fit_begin: ', fit_begin)
            model.fit(x_train_vec, y_train_vec)
            fit_end = time.time()
            fit_duration = fit_end - fit_begin
            log('fit_end: ', fit_end, ' duration: ', fit_duration)
            stats_buffer.write(f'Fit Duration: {fit_duration:3f}\n')
        
            # validation set
            eval(model, x_val_vec, y_val_vec, reduce_labels_rate, test=False)
            
            # test set
            eval(model, x_test_vec, y_test_vec, reduce_labels_rate, test=True)
            stats_buffer.write(f'--------------------------------------------------\n\n')
    

def main():
    setSeed()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='arxiv_data.json', nargs='?',
                        help="Path to the training data JSON file")
    parser.add_argument("--output", type=str, default='results.txt', nargs='?',
                        help="Path to the output text file")
    parser.add_argument("--reduce_labels_rate", type=int, default=0, nargs='?',
                        help="Labels percentage to retain")
    parser.add_argument("--debug", type=bool, default=False, nargs='?',
                        help="Debug model enable print log")
    args = parser.parse_args()
    train_file = args.data
    output_file = args.output
    reduce_labels_rate = args.reduce_labels_rate
    os.environ['DEBUG_MODE'] = str(args.debug)
    x_train_vec, x_val_vec, x_test_vec, y_train_vec, y_val_vec, y_test_vec = prepareData(train_file, reduce_labels_rate)
    trainAndEval(x_train_vec, x_val_vec, x_test_vec, y_train_vec, y_val_vec, y_test_vec, reduce_labels_rate)
    with open(output_file, "w") as results:
        results.write(stats_buffer.getvalue())
    

if __name__ == "__main__":
    main()