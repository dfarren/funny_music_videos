from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
import numpy as np
import pdb
import dill
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt



def load_word_vectors(path):

    with open(path, 'r') as f:
        vectors = f.readlines()

    embeddings = Counter()
    for i, vec in enumerate(vectors):
        embeddings[vec.split()[0]] = np.array([float(x) for x in vec.split()[1:]])
        if i%100000 == 0:
            print("%d vectors loaded" % i)

    return embeddings


def load_data(path='videos_data'):
    with open('videos_data', 'rb') as f:
        data = dill.load(f)
    return data


def clean_string(string):

    stop = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    out = []
    for word in tokenizer.tokenize(string):
        if word not in stop:
            out.append(word.lower())

    return out


def word_vector_from_text(string, word_vectors):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    :param x: string
    :param feature_dct: dict we use to update or append new features
    :param kgram_size:
    :return feature_dct: dict with feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """

    words = clean_string(string)
    vec_sum = np.zeros((len(word_vectors.values()[0]),), dtype=float)
    for word in words:
        vec_sum += word_vectors[word.encode('utf-8')]

    return vec_sum, len(words)


def word_vectors_from_video(video_data, word_vectors):
    """
    :param video_data: dict with video videos_data
    :return:
    """

    title_vector, title_n = word_vector_from_text(video_data['title'], word_vectors)
    description_vector, description_n = word_vector_from_text(video_data['description'], word_vectors)

    comments_vector = np.zeros((len(word_vectors.values()[0]),), dtype=float)
    comment_n = 0
    reply_vector = np.zeros((len(word_vectors.values()[0]),), dtype=float)
    reply_n = 0
    for comment in video_data['comments'].itervalues():
        comm_vec, comm_n = word_vector_from_text(comment['text'], word_vectors)
        comments_vector += comm_vec
        comment_n += comm_n
        if 'replies' in comment:
            for reply in comment['replies'].itervalues():
                rep_vec, rep_n = word_vector_from_text(reply['text'], word_vectors)
                reply_vector += rep_vec
                reply_n += rep_n

    tags_vector = np.zeros((len(word_vectors.values()[0]),), dtype=float)
    for tag in video_data['tags']:
        tags_vector += word_vectors[tag.encode('utf-8')]

    title_avg = title_vector / title_n
    description_avg = description_vector / description_n
    comments_avg = (comments_vector+reply_vector) / (comment_n+reply_n)
    tags_avg = tags_vector / len(video_data['tags'])

    return title_avg, description_avg, comments_avg, tags_avg


def calculate_training_vectors(data, word_vectors):

    videos_id = []
    X_title = []
    X_description = []
    X_comments = []
    X_tags = []
    y = []

    print("loading training vectors")
    print("nbr of vectors to load: %d" % (len(data['comedy'])+len(data['no_comedy'])))
    m = 0
    for category, label in [('comedy', 1), ('no_comedy', 0)]:
        for key, video in data[category].iteritems():

            # don't load videos wih comments disabled or without tags
            if 'comments' not in video or len(video['comments']) == 0 or \
                'tags' not in video or len(video['tags']) == 0 or \
                'title' not in video or len(video['title']) == 0 or \
                'description' not in video or len(video['description']) \
                    == 0:
                continue

            videos_id.append(key)

            title_avg, description_avg, comments_avg, tags_avg = word_vectors_from_video(video, word_vectors)
            X_title.append(title_avg)
            X_description.append(description_avg)
            X_comments.append(comments_avg)
            X_tags.append(tags_avg)
            y.append(label)

            m += 1
            if m%10==0:
                print("loaded %d training vectors" % m)

    perm = np.random.permutation(m)

    videos_id = np.array(videos_id)[perm]
    X_title = np.array(X_title)[perm]
    X_description = np.array(X_description)[perm]
    X_comments = np.array(X_comments)[perm]
    X_tags = np.array(X_tags)[perm]
    y = np.array(y)[perm]

    return videos_id, X_title, X_description, X_comments, X_tags, y


def calculate_testing_vectors(data, word_vectors):

    videos_id = []
    X_title = []
    X_description = []
    X_comments = []
    X_tags = []

    print("loading testing vectors")
    print("nbr of vectors to load: %d" % len(data['music']))

    m=0
    for key, video in data['music'].iteritems():

        # don't load videos wih comments disabled or without tags
        if 'comments' not in video or len(video['comments']) == 0 or \
            'tags' not in video or len(video['tags']) == 0 or \
            'title' not in video or len(video['title']) == 0 or \
            'description' not in video or len(video['description']) \
                == 0:
            continue

        videos_id.append(key)

        title_avg, description_avg, comments_avg, tags_avg = word_vectors_from_video(video, word_vectors)
        X_title.append(title_avg)
        X_description.append(description_avg)
        X_comments.append(comments_avg)
        X_tags.append(tags_avg)

        m += 1
        if m%10==0:
            print("loaded %d testing vectors" % m)

    perm = np.random.permutation(m)

    videos_id = np.array(videos_id)[perm]
    X_title = np.array(X_title)[perm]
    X_description = np.array(X_description)[perm]
    X_comments = np.array(X_comments)[perm]
    X_tags = np.array(X_tags)[perm]

    return videos_id, X_title, X_description, X_comments, X_tags


def save_(obj):
    name = [name for name in globals() if globals()[name] is obj][0]
    with open('data/'+name, 'wb') as f:
        dill.dump(obj, f)


def save(obj):
    if type(obj) is list:
        for e in obj:
            save_(e)
    else:
        save(obj)


def load_video_vectors(path, vec_list):
    out = []
    for vec in vec_list:
        with open(path+'/'+vec, 'rb') as f:
            vector = dill.load(f)
        out.append(vector)

    return out



def train_model(X_train, X_val, y_train, y_val):

    model = LogisticRegression()

    train_sizes, train_scores, valid_scores = learning_curve(model, X_train, y_train, cv=5)

    print train_sizes
    print train_scores
    print valid_scores
    #model.fit(X_train, y_train)
    #y_pred = model.predict(X_val)
    pdb.set_trace()
    #print("acuracy of model %f" % (float(np.sum(y_pred == y_val))/y_val.shape[0]))

    return model


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


def split_data(vector, validation_size):

    train = vector[:int(validation_size*vector.shape[0])]
    validation = vector[int(validation_size*vector.shape[0]):]

    return train, validation


def main():

    # load data, create training vectors, and save them
    embeddings = load_word_vectors('word_vectors/glove.6B.50d.txt')
    data = load_data('videos_data')
    #videos_id, X_title, X_description, X_comments, X_tags, y = calculate_training_vectors(data, embeddings)
    #save([videos_id, X_title, X_description, X_comments, X_tags, y])

    # load training vectors
    videos_id, X_title, X_description, X_comments, X_tags, y = \
        load_video_vectors('data', ['videos_id', 'X_title', 'X_description', 'X_comments', 'X_tags', 'y'])


    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    model_comments = MLPClassifier(max_iter=500, hidden_layer_sizes=(100, 100))
    plt = plot_learning_curve(model_comments, 'comments', X_comments, y, ylim=(0.0, 1.01), cv=cv, n_jobs=4)

    plt.show()

    pdb.set_trace()







    X_comments_train, X_comments_validation = split_data(X_comments, 0.7)
    y_train, y_validation = split_data(y, 0.7)


    model_comments = train_model(X_comments_train, X_comments_validation, y_train, y_validation)
    save(model_comments)

    videos_id, X_title, X_description, X_comments, X_tags = calculate_testing_vectors(data, embeddings)

    X_pred = model_comments.predict_proba(X_comments)

    idx = X_pred.argsort()
    print(X_pred[idx[::-1]][:10])
    print(videos_id[idx[::-1]][:10])



main()

