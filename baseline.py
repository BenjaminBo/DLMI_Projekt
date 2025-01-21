import cv2
import numpy as np
from PIL import Image
import glob
from typing import List, Any
import pandas as pd
import scipy
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import SVC

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


# bleeding = 1, healthy = 0


folder_bleeding = "data/train/bleeding"
folder_healthy = "data/train/healthy"

folder_test_healthy = "data/test/healthy"
folder_test_bleeding = "data/test/bleeding"

def load_images(folder_healthy, folder_bleeding):
    # get all images from folder
    healthy_images = [cv2.imread(file) for file in glob.glob(folder_healthy + "/*.jpg")]
    bleeding_images = [cv2.imread(file) for file in glob.glob(folder_bleeding + "/*.jpg")]
    return healthy_images, bleeding_images

def image_analysis(healthy_images:List[Any], bleeding_images: List[Any]):
    h_statistics_list_healthy = []
    h_statistics_list_bleeding = []
    [h_statistics_list_healthy.append(analyze_h_component_statistics(img)) for img in healthy_images]
    [h_statistics_list_bleeding.append(analyze_h_component_statistics(img)) for img in bleeding_images]
   
    healthy_df = pd.DataFrame(h_statistics_list_healthy)
    bleeding_df = pd.DataFrame(h_statistics_list_bleeding)
    print(healthy_df.describe())
    print(bleeding_df.describe())


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


# extract h component
def extract_h_features(img):
    img_hsv =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = img_hsv[:, :, 0] # np.array
    return np.array([np.mean(h), np.median(h),scipy.stats.mode(h.flatten())[0], np.sum(h)])
   
def test(folder_test_healthy, folder_test_bleeding, model):
    # collect test data
    healthy_images, bleeding_images = load_images(folder_test_healthy, folder_test_bleeding)
    X_test, y_test = build_data_matrix(healthy_images, bleeding_images)
    print(len(X_test))
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

def linear_regression():  
    healthy_images, bleeding_images = load_images(folder_healthy, folder_bleeding)
    X, y = build_data_matrix(healthy_images, bleeding_images)
    #weighted_y = compute_sample_weight(class_weight="balanced", y=y)
    model = LogisticRegression()
    model.fit(X, y)
    #model.fit(X,y, sample_weight=compute_class_weights(y))
    #model.fit(X, weighted_y)
    test(folder_test_healthy, folder_test_bleeding, model)

def compute_class_weights(y):
    class_counts = np.bincount(y.astype(int))
    class_weights = {cls: 1.0 / count for cls, count in enumerate(class_counts)}
    sample_weights = np.array([class_weights[cls] for cls in y.astype(int)])

    return sample_weights

def svm():
    healthy_images, bleeding_images = load_images(folder_healthy, folder_bleeding)
    X, y = build_data_matrix(healthy_images, bleeding_images)
    print("built data matrix")
    kernels = [ 'poly', 'sigmoid'] #'rbf',
    for k in kernels:
        print(f"using kernel {k}")
        svm_model = SVC(kernel=k)
        svm_model.fit(X, y)
        print("fitted model")
        test(folder_test_healthy, folder_test_bleeding, svm_model)


def evaluate_classifiers():
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
    
    healthy_images, bleeding_images = load_images(folder_healthy, folder_bleeding)
    X, y = build_data_matrix(healthy_images, bleeding_images)
    for name, clf in zip(names, classifiers):
        print(f"evaluating {name}")
        clf.fit(X, y)
        test(folder_test_healthy, folder_test_bleeding, clf)
        print("--------------------------")

evaluate_classifiers()