from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import VotingClassifier
import numpy as np
import pdb
import dill
import matplotlib.pyplot as plt
import web


def load_object(path):
    with open(path, 'rb') as f:
        data = dill.load(f)
    return data


def save_(obj):
    name = [name for name in globals() if globals()[name] is obj][0]
    with open('data/' + name, 'wb') as f:
        dill.dump(obj, f)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    return plt


def plot_learning_curves(X, y, model='NN'):

    if model == 'NN':
        model = MLPClassifier(verbose=True, max_iter=30, early_stopping=False)
    elif model == 'LR':
        model = LogisticRegression(C=1.5)
    elif model == 'SVM':
        model = SVC(kernel='linear')
    elif model== 'DT':
        model = DecisionTreeClassifier()
    elif model == 'KNN':
        model = KNeighborsClassifier()

    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    plt = plot_learning_curve(model, 'comments', X, y, ylim=(0.0, 1.01), cv=cv, n_jobs=4)

    return plt


def train(X, y, validation_size=0.2, model='NN'):

    print("training...")

    if model == 'NN':
        model = MLPClassifier(verbose=True, max_iter=30, early_stopping=False)
    elif model == 'LR':
        model = LogisticRegression(C=1.5)
    elif model == 'SVM':
        model = SVC(kernel='linear', probability=True)
    elif model== 'DT':
        model = DecisionTreeClassifier()
    elif model == 'KNN':
        model = KNeighborsClassifier()

    X_t = X[:int((1-validation_size)*X.shape[0])]
    y_t = y[:int((1-validation_size)*y.shape[0])]
    X_v = X[int((1-validation_size)*X.shape[0]):]
    y_v = y[int((1-validation_size)*y.shape[0]):]

    model.fit(X_t, y_t)

    print("accuracy: %f" % (float(np.sum(model.predict(X_v) == y_v))/y_v.shape[0]))

    print("done training")
    return model


def create_training_corpus(data):

    videos_id = []
    corpus = []
    tags = []
    stats = []
    labels = []

    m = 0
    print("will create %d documents" % (len(data['comedy'])+len(data['no_comedy'])))

    for category, label in [('comedy', 1), ('no_comedy', 0)]:

        for key, video in data[category].iteritems():

            string = video['title'] + '\n' + video['description']

            if 'comments' not in video or len(video['comments']) == 0 or 'tags' not in video or len(video['tags']) == 0 or \
               'title' not in video or len(video['title']) == 0 or 'description' not in video or len(video['description']) == 0:
                continue

            for comment in video['comments'].itervalues():
                string += comment['text']
                if 'replies' in comment:
                    for reply in comment['replies'].itervalues():
                        string += reply['text']

            tags_in_this_video = {}
            for tag in video['tags']:
                tags_in_this_video[tag.encode('utf-8')] = 1

            stats_in_this_video = {}
            for stat, cnt in video['statistics'].iteritems():
                stats_in_this_video[stat.encode('utf-8')] = int(cnt)

            videos_id.append(key)
            corpus.append(string)
            tags.append(tags_in_this_video)
            stats.append(stats_in_this_video)
            labels.append(label)

            m += 1

    print("done creating documents")

    perm = np.random.permutation(m)

    videos_id = np.array(videos_id)[perm]
    corpus = np.array(corpus)[perm]
    tags = np.array(tags)[perm]
    stats = np.array(stats)[perm]
    labels = np.array(labels)[perm]

    return videos_id, corpus, tags, stats, labels


def create_testing_corpus(data):

    videos_id = []
    corpus = []
    tags = []
    stats = []

    m = 0
    category = 'music'
    print("will create %d documents" % len(data[category]))

    for key, video in data[category].iteritems():

        string = video['title'] + '\n' + video['description']

        if 'comments' not in video or len(video['comments']) == 0 or 'tags' not in video or len(video['tags']) == 0 or \
           'title' not in video or len(video['title']) == 0 or 'description' not in video or len(video['description']) == 0:
            continue

        for comment in video['comments'].itervalues():
            string += comment['text']
            if 'replies' in comment:
                for reply in comment['replies'].itervalues():
                    string += reply['text']

        tags_in_this_video = {}
        for tag in video['tags']:
            tags_in_this_video[tag.encode('utf-8')] = 1

        stats_in_this_video = {}
        for stat, cnt in video['statistics'].iteritems():
            stats_in_this_video[stat.encode('utf-8')] = int(cnt)

        videos_id.append(key)
        corpus.append(string)
        tags.append(tags_in_this_video)
        stats.append(stats_in_this_video)

        m += 1

    print("done creating documents")

    perm = np.random.permutation(m)

    videos_id = np.array(videos_id)[perm]
    corpus = np.array(corpus)[perm]
    tags = np.array(tags)[perm]
    stats = np.array(stats)[perm]

    return videos_id, corpus, tags, stats


def main(nbr_videos):

    data = load_object('videos_data')

    # create corpus: titles, comments, and replies from all videos
    videos_id, corpus, tags, stats, labels = create_training_corpus(data)

    # create tf-idf vectors
    vectorizer_tfidf = TfidfVectorizer(stop_words='english')
    tf_idf_train = vectorizer_tfidf.fit_transform(corpus)

    # create tag matrix
    vectorizer_tags = DictVectorizer()
    tags_train = vectorizer_tags.fit_transform(tags)

    # train NN on titles, comments, and replies
    model_text = train(tf_idf_train, labels, model='SVM')
    model_tags = train(tags_train, labels, model='LR')

    #
    videos_id, corpus, tags, stats = create_testing_corpus(data)
    tf_idf_test = vectorizer_tfidf.transform(corpus)
    tags_test = vectorizer_tags.transform(tags)

    labels_predicted_by_text = model_text.predict_proba(tf_idf_test)
    labels_predicted_by_tags = model_tags.predict_proba(tags_test)

    labels_pred = np.mean([labels_predicted_by_text, labels_predicted_by_tags], axis=0)
    labels_pred = labels_pred[:, 1]

    idx = labels_pred.argsort()

    funny_music_videos = []
    for video_id in videos_id[idx[::-1]][:nbr_videos]:
        funny_music_videos.append(video_id)

    web.add_videos(funny_music_videos)

    #print(labels_pred[idx[::-1]][:20])
    #print(videos_id[idx[::-1]][:20])
