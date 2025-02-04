import cv2
import numpy as np
from PIL import Image
import glob
from typing import List, Any
import pandas as pd
import scipy
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
# bleeding = 1, healthy = 0


folder_bleeding = "project_capsule_dataset_for_classifiacation/project_capsule_dataset_for_classifiacation/train_equal_instances/bleeding"
folder_healthy = "project_capsule_dataset_for_classifiacation/project_capsule_dataset_for_classifiacation/train_equal_instances/healthy"

folder_test_healthy = "project_capsule_dataset_for_classifiacation/project_capsule_dataset_for_classifiacation/test/healthy"
folder_test_bleeding = "project_capsule_dataset_for_classifiacation/project_capsule_dataset_for_classifiacation/test/bleeding"

def load_images(folder_healthy:str, folder_bleeding:str):
    # get all images from folder
    healthy_images = [cv2.imread(file) for file in glob.glob(folder_healthy + "/*.jpg")]
    bleeding_images = [cv2.imread(file) for file in glob.glob(folder_bleeding + "/*.jpg")]
    return healthy_images, bleeding_images

def build_data_matrix(healthy_images: List[Any], bleeding_images: List[Any]):
    x_healthy_features = []
    if len(healthy_images) > 570:
        healthy_images = healthy_images[:570]
    for idx, img in enumerate(healthy_images[:570]):
        x_healthy_features.append([extract_h_features(img)])# np.array
    healthy_features_array = np.concatenate(x_healthy_features)
    healthy_labels = np.zeros(healthy_features_array.shape[0])

    x_bleeding_features = []
    for idx, img in enumerate(bleeding_images):
        x_bleeding_features.append([extract_h_features(img)])
    bleeding_features_array = np.concatenate(x_bleeding_features)
    bleeding_labels = np.ones(bleeding_features_array.shape[0])

    return np.concatenate((healthy_features_array, bleeding_features_array)), np.concatenate((healthy_labels, bleeding_labels))


def get_train_data():
    healthy_images, bleeding_images = load_images(folder_healthy, folder_bleeding)
    X, y = build_data_matrix(healthy_images, bleeding_images)
    return X,y

def get_test_data():
    healthy_images_test, bleeding_images_test = load_images(folder_test_healthy, folder_test_bleeding)
    X_test, y_test = build_data_matrix(healthy_images_test, bleeding_images_test)
    return X_test, y_test
   
def extract_h_features(img):
    img_hsv =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = img_hsv[:, :, 0] # np.array
    return np.array([np.mean(h), np.median(h),scipy.stats.mode(h.flatten())[0], np.sum(h)])
   
def test(model):
    X_test, y_test = get_test_data()
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))


def compute_class_weights(y):
    class_counts = np.bincount(y.astype(int))
    class_weights = {cls: 1.0 / count for cls, count in enumerate(class_counts)}
    sample_weights = np.array([class_weights[cls] for cls in y.astype(int)])
    return sample_weights

def create_model(name:str):
    names = [
        "nearest_neighbors",
        "svm",
        "gaussian_process",
        "decision_tree",
        "random_forest",
        "neural_net",
        "adaboost",
        "naive_bayes",
        "qda",
        "logistic_regression",
    ]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(gamma=2, C=1, random_state=42),
        GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1, random_state=42
        ),
        MLPClassifier(alpha=1, max_iter=1000, random_state=42),
        AdaBoostClassifier(random_state=42),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        LogisticRegression()
    ]
    classifier_dict = dict(zip(names, classifiers))
    X, y = get_train_data()
    print("built data matrix")
    model = classifier_dict.get(name)
    if model is None:
        raise ValueError("Invalid model name")
    model.fit(X, y)
    print("fitted model")
    test(model)
    joblib.dump(model, f"{name}_baseline.pkl")


def evaluate_different_classifiers():
    names = [
        "Nearest Neighbors",
        "rbf SVM with C=0.025",
        "RBF SVM with C =1",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="rbf", C=0.025, random_state=42),
        SVC(gamma=2, C=1, random_state=42),
        GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1, random_state=42
        ),
        MLPClassifier(alpha=1, max_iter=1000, random_state=42),
        AdaBoostClassifier(random_state=42),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]
    X, y = get_train_data()
    for name, clf in zip(names, classifiers):
        print(f"evaluating {name}")
        clf.fit(X, y)
        test(clf)
        print("--------------------------")


def load_and_test(path_model):
    model = joblib.load(path_model)
    test(model)


if __name__ == "__main__":
    create_model("random_forest")
    print("created model")
    load_and_test("random_forest_baseline.pkl")
