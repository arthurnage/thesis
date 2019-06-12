from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


class DetectorClassifier(BaseEstimator):
    """
    Classifier that adapts the inner classifier to concept drift changes.
    """
    def __init__(self, clf, detection_method, classes):
        if not hasattr(clf, "partial_fit"):
            raise TypeError("Choose incremental classifier")
        self.clf = clf
        self.detection_method = detection_method
        self.classes = classes
        self.history = []
        self.change_detected = 0

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def partial_fit(self, X, y):
        pre_y = self.clf.predict(X)
        score = accuracy_score(pre_y, y)
        self.history.append(score)
        
        if self.detection_method.set_input(score):
            self.change_detected += 1
            self.clf = clone(self.clf)
            self.clf.partial_fit(X, y, classes=self.classes)
            return True
        else:
            self.clf.partial_fit(X, y)
            return False

    def predict(self, X):
        return self.clf.predict(X)
