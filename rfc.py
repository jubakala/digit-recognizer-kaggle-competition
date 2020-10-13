import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

"""
These are the machine learning models I used to participate Kaggle's Digit Recognizer competition (https://www.kaggle.com/c/digit-recognizer).
I tried these before using convolutional neural network. The best results were achieved using Random forest classifier. It reached the 
accuracy of 0.964. I also tested SVC and K-neightbors classifiers, but their results werent' as good as RFC's.
Best results:
    - Random forest classifier 0.964
    - SVC 0.921
    - K-nearest neighbors 0.936
"""

def preds_to_file(test_all, preds, filename):
    output = pd.DataFrame({'Label': preds})
    output.to_csv("results/" + filename, index=False)
    print("Results saved!")


def get_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    test_all = pd.read_csv('data/test.csv') # This is used only to get the correct ids to the predictions file.

    return train, test, test_all


def standardize(X_train, X_test, test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    test = sc.transform(test)

    return X_train, X_test, test


def get_svc():
    return SVC(kernel = 'linear', random_state = 0)


def get_kneighbors(neighbors_count):
    return KNeighborsClassifier(n_neighbors=neighbors_count, metric='minkowski', p=2)


def get_rfc(estimators_count):
    return RandomForestClassifier(n_estimators=estimators_count, criterion='entropy', random_state=0)


def test_model(classifier, X_test, y_test, model):
    y_pred = classifier.predict(X_test)
    print(accuracy_score(y_test, y_pred))


def main():
    models = ["rfc", "svc", "k-neighbors"]
    train, test, test_all = get_data()    
    
    X = train.iloc[:, 1:].values    
    y = train.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    X_train, X_test, test = standardize(X_train, X_test, test)
    
    # Testing different models.
    for model_name in models:
        print(model_name)        
        if model_name == "k-neighbors":
            classifier = get_kneighbors(10)
        if model_name == "rfc":
            classifier = get_rfc(300) 
        if model_name == "svc":
            classifier = get_svc()

        # Fit the model.
        print("Starting the training...")
        classifier.fit(X_train, y_train)
        # Test the model.
        print("Testing the model...")
        test_model(classifier, X_test, y_test, model_name)

        # Make predictions.
        print("Making predictions...")
        predictions = classifier.predict(test)
        # Save results into a file.
        preds_to_file(test_all, predictions, "digits_" + model_name + ".csv")


if __name__ == "__main__":
    main()