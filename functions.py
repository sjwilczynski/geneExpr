import numpy as np
import itertools
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import rpy2.robjects as robjects
import os
import matplotlib.pyplot as plt

class_names = np.array(["No event", "Met event"])


def load_file(name):
    return np.genfromtxt(name, delimiter=",", skip_header=1)
def load_train_and_test_parts():
    X_train = load_file("data/microarray_train.csv")
    X_test = load_file("data/microarray_test.csv")
    y_train = load_file("data/labels_train.csv")
    y_test = load_file("data/labels_test.csv")
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(axis, cm, classes, 
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = axis.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    axis.set(title=title, xlabel='Predicted label', ylabel='True label')
    tick_marks = np.arange(len(classes))
    axis.set_xticks(tick_marks)
    axis.set_xticklabels(classes)
    axis.set_yticks(tick_marks)
    axis.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axis.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    return im


def fit_clf(clf, train_set, train_labels, test_set, test_labels):
    clf = clf.fit(train_set, train_labels)
    print("Train and test scores: {} {}".format(clf.score(train_set, train_labels), clf.score(test_set, test_labels)))
    plot_clf(clf, train_set, train_labels, test_set, test_labels)
    return clf

def plot_clf(clf, train_set, train_labels, test_set, test_labels):
    test_labels_pred = clf.predict(test_set)
    train_labels_pred = clf.predict(train_set)
    test_cm = confusion_matrix(test_labels, test_labels_pred)
    train_cm = confusion_matrix(train_labels, train_labels_pred)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True) 
    im = plot_confusion_matrix(ax1, test_cm, classes=class_names, normalize=True, title='Test confusion matrix')
    im = plot_confusion_matrix(ax2, train_cm, classes=class_names, normalize=True, title="Train confusion matrix")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
    
def fit_models(train_set, train_labels, test_set, test_labels):
    clf_logit = fit_clf(LogisticRegression(solver='liblinear', penalty='l1', C=0.3), train_set, train_labels, test_set, test_labels)
    plt.figure()
    plt.title('Logistics regression coefficients')
    plt.plot(np.arange(clf_logit.coef_.shape[1]), clf_logit.coef_[0])
    plt.show()
    clf_svm = fit_clf(SVC(gamma='scale', C=7), train_set, train_labels, test_set, test_labels)
    clf_tree = fit_clf(DecisionTreeClassifier(max_depth=4, min_samples_leaf=10), train_set, train_labels, test_set, test_labels)
    clf_forest = fit_clf(RandomForestClassifier(max_depth=4, n_estimators=100, min_samples_leaf=10), train_set, train_labels, test_set, test_labels)
    return (clf_logit, clf_svm, clf_tree, clf_forest)


## MLCC
def read_mlcc_result(filename, train_size):
    robjects.r['load']("./mlcc_results/{}".format(filename))
    s, m, b = robjects.r['res']
    segmentation = np.asarray(s)
    numb_clust = np.max(s)
    mBIC = np.asarray(m)
    b.names = robjects.r('0:{}'.format(numb_clust-1))
    bases = dict(zip(b.names, map(list,list(b))))
    dimensionalities = np.empty(numb_clust, dtype=np.int32)
    for i in range(numb_clust):
        dimensionalities[i] = len(bases[str(i)]) // train_size
    return segmentation-1, mBIC, dimensionalities

def apply_mlcc_dim_reduction(X, segmentation, dimensionalities):
    numb_clust = dimensionalities.shape[0]
    X_reduced = np.empty((X.shape[0], 0))
    for i in range(numb_clust):
        cluster = X[:, segmentation == i]
        n_components = dimensionalities[i]
        if cluster.shape[1] < n_components: #TODO - maybe mlcc shouldn't allow it
            print("WARNING! Dimensionality of a cluster was greater than the number of variables. Ignoring this cluster.")
        else:
            X_reduced = np.concatenate((X_reduced, PCA(n_components=n_components).fit_transform(cluster)), axis=1)
    return X_reduced