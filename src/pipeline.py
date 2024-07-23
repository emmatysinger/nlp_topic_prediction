import argparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import GridSearchCV
import src.explainability as explainability
import joblib

def prepare_target(bunch, binary_target):
    target = np.zeros(len(bunch.target), dtype=bool)
    for binary_target_ in binary_target:
        binary_target_idx = bunch.target_names.index(binary_target_)
        target |= (bunch.target == binary_target_idx)
    return target

def get_metrics(targets, predictions, confusion=False):
    print(f"Accuracy of our classifier is: {accuracy_score(targets, predictions)}")
    metrics = precision_recall_fscore_support(targets, predictions)
    print(f"Precision of our classifier for positive labels is: {metrics[0][1]}")
    print(f"Recall of our classifier for positive labels is: {metrics[1][1]}")
    print(f"F1 Score of our classifier for positive labels is: {metrics[2][1]}")

    if confusion:
        disp = ConfusionMatrixDisplay(confusion_matrix(target_test, pred_test))
        disp.plot()

def hyperparameter_tuning(clf, param_grid):
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1')
    print("Fitting the grid search pipeline...")
    grid_search.fit(train_X, target_train)
    print("Fitting the grid search pipeline... Done")

    # Cross-validation score for tuned pipeline
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Tuned cross-validation f1 score: {grid_search.best_score_:.4f}")

    # Evaluate on the test set
    best_clf = grid_search.best_estimator_
    pred_test = best_clf.predict(test_X)

    return best_clf, pred_test


if __name__ == "__main__":
    # TODO: expose a nice CLI
    parser = argparse.ArgumentParser(description="Text classifier.")
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--explain', action='store_true', help='Look into the explainability of the model')
    parser.add_argument('--hyperparameter', action='store_true', help='Hyperparameter tuning')
    parser.add_argument('--classifier', type=str, choices=['decision_tree', 'random_forest'], default='decision_tree', help='Classifier to use')
    args = parser.parse_args()

    NEWSGROUP_DOWNLOAD_PATH = "./"
    binary_target = "talk.religion.misc", "alt.atheism", "soc.religion.christian"
    random_state = 123
    min_token_document_frequency = 10

    newsgroup_bunch_train = fetch_20newsgroups(
        data_home=NEWSGROUP_DOWNLOAD_PATH,
        random_state=random_state,
        subset="train",
        remove=("headers", "footers", "quotes"),
    )
    newsgroup_bunch_val = fetch_20newsgroups(
        data_home=NEWSGROUP_DOWNLOAD_PATH,
        random_state=random_state,
        subset="test",
        remove=("headers", "footers", "quotes"),
    )

    # Remove stop words and use n-grams (to add more context to the model)
    tfidf_transformer = TfidfVectorizer(
        min_df=min_token_document_frequency,
        stop_words='english',
        ngram_range=(1, 2)
    )

    # Added a scalar and Pipeline
    transform_pipeline = Pipeline([('tfidf', tfidf_transformer), ('scaler', StandardScaler(with_mean=False))]) #use with_mean=False because it's a sparse matrix

    print("Fitting my transform pipeline...")
    transform_pipeline.fit(newsgroup_bunch_train.data)
    print("Fitting my transform pipeline... Done")

    print("Transforming train and test splits into tfidf values...")
    train_X = transform_pipeline.transform(newsgroup_bunch_train.data)
    test_X = transform_pipeline.transform(newsgroup_bunch_val.data)
    print("Transforming train and test splits into tfidf values... Done")


    # Prepare the target
    # TODO: factorize instead of copy/pasting for train and test
    
    
    target_train = prepare_target(newsgroup_bunch_train, binary_target)
    target_test = prepare_target(newsgroup_bunch_val, binary_target)

    if args.classifier == 'decision_tree':
        clf = DecisionTreeClassifier(random_state=random_state, max_depth=20, min_samples_split=5)
        param_grid_clf = {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    elif args.classifier == 'random_forest':
        clf = RandomForestClassifier(random_state=random_state)
        param_grid_clf = {
            'n_estimators': [2,10,50,100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    else:
        raise ValueError('Classifier must either be decision_tree or random_forest')

    if args.hyperparameter:
        best_clf, pred_test = hyperparameter_tuning(clf, param_grid=param_grid_clf)

    else:
        print(f"Train {clf} classifier...")
        pred_train = clf.fit(train_X, target_train).predict(train_X)
        print(f"Train {clf} classifier... Done.")
        pred_test = clf.predict(test_X)

    if args.evaluate:
        get_metrics(target_test, pred_test, confusion=True)

    if args.explain:
        feature_names = transform_pipeline.named_steps['tfidf'].get_feature_names_out()
        positive_indices = [i for i in range(len(pred_test)) if pred_test[i]]
        random_sample_indices = random.sample(positive_indices, 5)

        # Process and display important features for each sampled index
        for sample_index in random_sample_indices:
            sample_data = test_X[sample_index]
            print(f"Important features for sample: {newsgroup_bunch_val.data[sample_index]}")
            explanations = explainability.explain_prediction(sample_data, clf, feature_names)
            for feature, importance in explanations[:10]:  # Display top 10 features
                print(f"Feature: {feature}, Importance: {importance:.4f}")

            print('\n')

    joblib.dump(clf, 'model.joblib')
    joblib.dump(transform_pipeline, 'transform_pipeline.joblib')
