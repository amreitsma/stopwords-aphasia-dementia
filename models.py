from TFIDF import *
from comparison import *
import matplotlib.pyplot as plt
from sklearn import utils
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support, \
    confusion_matrix, ConfusionMatrixDisplay


def shuff_data(feats, labels):
    """Shuffle the data randomly with seed"""
    return utils.shuffle(feats, labels, random_state=42)


def run_baseline(true_labels):
    """Print performance scores of a baseline predicting the most frequent class for each sample in the data set"""
    pred_labels = np.array(["AD" for i in range(277)])
    print("Accuracy of the 'most frequent class' baseline is: {}".format(accuracy_score(true_labels, pred_labels)))
    scores = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)
    print("Precision:\t{}\nRecall:\t\t{}\nF1-score:\t{}".format(scores[0], scores[1], scores[2]))


def cross_val(feats, labels, clf):
    """Print performance scores of a classifier averaged over 5-fold cross-validation."""
    scoring = ["accuracy", "f1_macro", "recall_macro", "precision_macro"]
    scores = cross_validate(clf, feats, labels, scoring=scoring)
    print("Scores for {0:25}:\taccuracy {1},\tf1"
          "score {2},\tprecision {3}\tand "
          "recall {4}".format(clf.__class__.__name__, sum(scores["test_accuracy"])/5, sum(scores["test_f1_macro"])/5,
                              sum(scores["test_precision_macro"])/5, sum(scores["test_recall_macro"])/5))


def cross_val_confusion(feats, labels, clf):
    """Obtain aggregated predictions over the full dataset after applying stratified k-fold for a given model (clf)"""
    k_fold = StratifiedKFold()
    pred_labels = np.array([])
    real_labels = np.array([])

    for train_id, test_id in k_fold.split(feats, labels):
        train_feat = feats[train_id]
        train_label = labels[train_id]
        test_feat = feats[test_id]
        test_label = labels[test_id]

        clf.fit(train_feat, train_label)
        preds = clf.predict(test_feat)
        pred_labels = np.append(pred_labels, preds)
        real_labels = np.append(real_labels, test_label)

    return pred_labels, real_labels


def draw_matrices(pred, real, mod_name):
    """Given predicted labels and true labels, plot a confusion matrix"""
    matrix = confusion_matrix(real, pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["AD", "EA"])
    cm_display.plot()
    cm_display.ax_.set_title("Confusion matrix for {} model".format(mod_name))
    plt.show()


def run_models(feats, labels):
    """Train and evaluate a predefined list of models using 5-fold cross-validation"""
    models = [SVC(class_weight="balanced"), LogisticRegression(class_weight="balanced"), ComplementNB(),
              DecisionTreeClassifier(random_state=42, class_weight="balanced")]
    for model in models:
        cross_val(feats, labels, model)
        pred, real = cross_val_confusion(feats, labels, model)
        draw_matrices(pred, real, model.__class__.__name__)
    print()


def calc_stopwds():
    """Calculate the average amount of stop words per 100 words for both groups."""
    average_stopwds("Dementia_chat.zip", "ProbableAD", True)
    average_stopwds("Aphasia_chat.zip", "Broca", True)


def main_experiment(labels, ad_data, ea_data, stop):
    """Run the experiment and return model scores plus confusion matrices"""
    feats = vectorize(list(ad_data) + ea_data, stop)
    feats, labels = shuff_data(feats, labels)
    run_models(feats, labels)


def stopwords_experiment(labels):
    """Run the experiment using only stop words as features and return model scores plus confusion matrices"""
    ad_data = average_stopwds("Dementia_chat.zip", "ProbableAD")
    ea_data = average_stopwds("Aphasia_chat.zip", "Broca")

    feats = vectorize(list(ad_data) + ea_data, max_ngram=1)
    feats, labels = shuff_data(feats, labels)
    run_models(feats, labels)


def main():
    np.random.seed(5)
    ad_data = preprocess_data("Dementia_chat.zip", "ProbableAD")
    ea_data = preprocess_data("Aphasia_chat.zip", "Broca")
    labels = np.array(["AD"]*234 + ["EA"]*43)
    # -------------------------------------------------------
    # Uncomment to calculate the average amount of stop words per 100 words
    # and show the most frequently used stop words:
    # calc_stopwds()
    # -------------------------------------------------------
    # Uncomment to run the baseline for the main experiment:
    # run_baseline(labels)
    # -------------------------------------------------------
    # Uncomment to run the main experiment with stop words:
    # main_experiment(labels, ad_data, ea_data, False)
    # Without stop words:
    # main_experiment(labels, ad_data, ea_data, True)
    # -------------------------------------------------------
    # Uncomment to run the main experiment on a balanced dataset with stop words:
    # NB: don't uncomment line 121 and line 123 at the same time, somehow this leads
    # to slightly different results for the complementNB model.
    # main_experiment(np.array(["AD"]*43 + ["EA"]*43), np.random.choice(ad_data, 43), ea_data, False)
    # Without stop words:
    # main_experiment(np.array(["AD"]*43 + ["EA"]*43), np.random.choice(ad_data, 43), ea_data, True)
    # -------------------------------------------------------
    # Uncomment to run the experiment on only the stop words:
    # stopwords_experiment(labels)


if __name__ == "__main__":
    main()
