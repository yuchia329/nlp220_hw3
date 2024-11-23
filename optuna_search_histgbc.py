import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from utils import setSeed
from loaddata import prepareData
from sklearn.metrics import precision_recall_fscore_support
from functools import partial


# Define the objective function for Optuna
def objective(trial, X_train, y_train, x_val_vec, y_val_vec):
    
    X_train = X_train.toarray()
    x_val_vec = x_val_vec.toarray()
    # Define the hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5)
    max_iter = trial.suggest_int("max_iter", 100, 300)
    max_depth = trial.suggest_int("max_depth", 10, 30)
    # l2_regularization = trial.suggest_loguniform('l2_regularization', 0.01, 0.5)
    # max_features = trial.suggest_loguniform('max_features', 0.5, 1)

    # Initialize the model with the suggested hyperparameters
    model = MultiOutputClassifier(HistGradientBoostingClassifier(learning_rate=learning_rate, max_iter=max_iter, max_depth=max_depth, random_state=42))
    # model = MultiOutputClassifier(HistGradientBoostingClassifier(learning_rate=learning_rate, max_iter=max_iter, max_depth=max_depth, l2_regularization=l2_regularization, max_features=max_features, random_state=42))

    # Use cross-validation to evaluate the model
    # score = cross_val_score(model, X_train, y_train, cv=3, scoring=make_scorer(f1_score, average="micro")).mean()
    # return score
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(x_val_vec)
    
    # Compute F1 score
    f1 = f1_score(y_val_vec, y_pred, average="micro")
    return f1


def main():
    train_file = 'arxiv_data.json'
    setSeed()
    x_train_vec, x_val_vec, x_test_vec, y_train_vec, y_val_vec, y_test_vec = prepareData(train_file)
    
    objective_with_params = partial(objective, X_train=x_train_vec, y_train=y_train_vec, x_val_vec=x_val_vec, y_val_vec=y_val_vec)

    # Run the Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_params, n_trials=50, n_jobs=-1)

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)
    print("Best F1 Score:", study.best_value)

    # Train the model on the full training data with the best parameters
    best_model = MultiOutputClassifier(HistGradientBoostingClassifier(**best_params, random_state=42))
    best_model.fit(x_val_vec, y_val_vec)

    # Evaluate the model on the test set
    y_pred = best_model.predict(x_test_vec)
    test_f1 = f1_score(y_test_vec, y_pred, average="micro")
    print("Test F1 Score:", test_f1)
    p, r, f1, _ = precision_recall_fscore_support(y_true=y_test_vec, y_pred=y_pred)
    print("precision: ", p, " recall: ", r, " f1_score: ", f1)

main()