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
def objective(trial, X_train, y_train):
    # Define the hyperparameters to tune
    n_estimators = trial.suggest_int("n_estimators", 10, 100)
    criterion = trial.suggest_categorical('criterion', ["gini", "entropy", "log_loss"])
    max_depth = trial.suggest_int("max_depth", 50, 150)
    max_features = trial.suggest_categorical('max_features', ["sqrt", "log2"])

    # Initialize the model with the suggested hyperparameters
    model = MultiOutputClassifier(ExtraTreesClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, max_features=max_features, n_jobs=-1, random_state=42))

    # Use cross-validation to evaluate the model
    score = cross_val_score(model, X_train, y_train, cv=3, scoring=make_scorer(f1_score, average="micro")).mean()
    return score


def main():
    train_file = 'arxiv_data.json'
    setSeed()
    x_train_vec, x_val_vec, x_test_vec, y_train_vec, y_val_vec, y_test_vec = prepareData(train_file)
    
    objective_with_params = partial(objective, X_train=x_val_vec, y_train=y_val_vec)

    # Run the Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_params, n_trials=50)

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    # Train the model on the full training data with the best parameters
    best_model = MultiOutputClassifier(ExtraTreesClassifier(**best_params, n_jobs=-1, random_state=42))
    best_model.fit(x_val_vec, y_val_vec)

    # Evaluate the model on the test set
    y_pred = best_model.predict(x_test_vec)
    test_f1 = f1_score(y_test_vec, y_pred, average="micro")
    print("Test F1 Score:", test_f1)
    p, r, f1, _ = precision_recall_fscore_support(y_true=y_test_vec, y_pred=y_pred)
    print("precision: ", p, " recall: ", r, " f1_score: ", f1)

main()