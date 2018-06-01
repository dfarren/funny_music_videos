from collections import Counter
import random
import math
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.sparse import coo_matrix
import numpy as np
import pdb
import dill
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def clean_string(string):

    stop = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    out = []
    for word in tokenizer.tokenize(string):
        if word not in stop:
            out.append(word.lower())

    return out


def extract_word_features_from_string(x, feature_dct, kgram_size=1):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    :param x: string
    :param feature_dct: dict we use to update or append new features
    :param kgram_size:
    :return feature_dct: dict with feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """

    words = clean_string(x)
    for i in range(len(words)-kgram_size+1):
        kgram = ''
        for j in range(kgram_size):
            kgram += words[i+j] + ' '
        kgram = 'kgram->'+kgram[:-1].encode('utf-8')

        feature_dct[kgram] += 1

    return feature_dct


def word_features_from_video(kgram_size=1):
    """
    :param video_data: dict with video videos_data
    :return:
    """

    def extract(video_data):

        feature_dct = Counter({'#!BIAS!#': 1})

        feature_dct = extract_word_features_from_string(video_data['title'], feature_dct, kgram_size)
        feature_dct = extract_word_features_from_string(video_data['description'], feature_dct, kgram_size)

        for comment in video_data['comments'].itervalues():
            feature_dct = extract_word_features_from_string(comment['text'], feature_dct, kgram_size)
            if 'replies' in comment:
                for reply in comment['replies'].itervalues():
                    feature_dct = extract_word_features_from_string(reply['text'], feature_dct, kgram_size)

        return feature_dct

    return extract


def tag_features_from_video(video_data):

    if 'tags' not in video_data:
        return {}

    feature_dct = Counter({'#!BIAS!#': 1})

    for tag in video_data['tags']:
        feature_dct['tag->'+tag.encode('utf-8')] = 1

    return feature_dct


def statistics_features_from_video(video_data):

    feature_dct = Counter({'#!BIAS!#': 1})

    for k, v in video_data['statistics'].iteritems():
        feature_dct['stat->'+k] = int(v)

    return feature_dct


def extract_features_from_all_videos(examples, feature_extractors, classifier='SVM'):

    out = []
    if type(examples[0]) is tuple:
        for video_data, y in examples:
            features_dct = {}
            for feature_extractor in feature_extractors:
                features_dct.update(feature_extractor(video_data))
            if len(features_dct) == 0:
                continue
            if y != 1:
                if classifier == 'SVM':
                    y = -1
                elif classifier == 'logistic':
                    y = 0
            out.append((features_dct, y))

    elif type(examples[0]) is dict:
        for video_data in examples:
            features_dct = {}
            for feature_extractor in feature_extractors:
                features_dct.update(feature_extractor(video_data))
            if len(features_dct) == 0:
                continue
            out.append(features_dct)
    else:
        raise ValueError("Examples have to be either a dict of features or a tuple with a dict of features as a label.\
                         I got this: %s" % str(examples[0]))
    return out


class SVM:

    def __init__(self, feature_extractors, regularization=None, num_epochs=20, alpha=0.03, lambda_=0.01):

        self.feature_extractors = feature_extractors
        self.regularization = regularization
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.lambda_ = lambda_
        self.weights = Counter({'#!BIAS!#': 1})  # feature => weight


    # stochastic gradient descent for a linear SVM
    def fit(self, train_examples, validation_examples):
        """
        Given |trainExamples| and |testExamples| (each one is a list of (x,y)
        pairs), a |featureExtractor| to apply to x, and the number of iterations to
        train |numIters|, return the weight vector (sparse feature vector) learned.

        You should implement stochastic gradient descent.

        Note: only use the trainExamples for training!
        You should call evaluatePredictor() on both trainExamples and testExamples
        to see how you're doing as you learn after each iteration.
        numIters refers to a variable you need to declare. It is not passed in.
        """

        # pre extract all examples
        train_examples = extract_features_from_all_videos(train_examples, self.feature_extractors)
        validation_examples = extract_features_from_all_videos(validation_examples, self.feature_extractors)

        epoch = 0
        while epoch < self.num_epochs:
            random.shuffle(train_examples)
            for feature_dct, y in train_examples:
                xw = 0
                for word, cnt in feature_dct.iteritems():
                    xw += self.weights[word] * cnt

                if y * xw < 1:
                    for word, cnt in feature_dct.iteritems():

                        if self.regularization == 'L1':
                            reg = math.copysign(1, self.weights[word])
                        elif self.regularization == 'L2':
                            reg = 2*self.weights[word]
                        else:
                            reg = 0

                        self.weights[word] += self.alpha * (y * cnt + self.lambda_*reg)

            # print "weights in trainer: %s" % weights
            print "epoch: %d    training error: %f    dev error: %f" % (epoch,
                                                                        evaluate_predictor(train_examples,
                                                                                           self.predict),
                                                                        evaluate_predictor(validation_examples,
                                                                                           self.predict))
            epoch += 1

        return

    def predict(self, feature_dct):

        xw = 0
        for word, cnt in feature_dct.iteritems():
            xw += self.weights[word] * cnt
        if xw > 0:
            return 1
        else:
            return -1


class Logistic_Regression:

    def __init__(self, name, feature_extractors, regularization=None, num_epochs=20, alpha=0.03, lambda_=0.01):
        self.name = name
        self.feature_extractors = feature_extractors
        self.regularization = regularization
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.lambda_ = lambda_
        self.weights = Counter({'#!BIAS!#': 1})  # feature => weight

    def fit(self, train_examples, validation_examples):

        # pre extract all examples
        train_examples = extract_features_from_all_videos(train_examples, self.feature_extractors,
                                                          classifier='logistic')

        validation_examples = extract_features_from_all_videos(validation_examples, self.feature_extractors,
                                                               classifier='logistic')

        #best_weights = {}
        #lowest_error = float('inf')
        epoch = 0
        while epoch < self.num_epochs:
            random.shuffle(train_examples)
            for feature_dct, y in train_examples:
                xw = 0
                gradients = {}
                # calculate gradients
                for word, cnt in feature_dct.iteritems():
                    xw += self.weights[word] * cnt

                    if self.regularization == 'L1':
                        reg = math.copysign(1, self.weights[word])
                    elif self.regularization == 'L2':
                        reg = 2*self.weights[word]
                    else:
                        reg = 0

                    gradients[word] = (self.logistic_fn(xw) - y) * cnt + self.lambda_*reg

                # update coefs
                for word, cnt in feature_dct.iteritems():
                    self.weights[word] -= self.alpha * gradients[word]

            training_error = evaluate_predictor(train_examples, self.predict_with_features)
            validation_error = evaluate_predictor(validation_examples, self.predict_with_features)
            print "epoch: %d    training error: %f    dev error: %f" % (epoch, training_error, validation_error)

            epoch += 1

        return

    def predict_one_example(self, feature_dct, probability=False):

        xw = 0
        for word, cnt in feature_dct.iteritems():
            xw += self.weights[word] * cnt

        if probability:
            return self.logistic_fn(xw)

        if self.logistic_fn(xw) > 0.5:
            return 1
        else:
            return 0

    def predict_with_features(self, features, probability=False):

        if type(features) is list:
            prediction = []
            for feature_dct in features:
                prediction.append(self.predict_one_example(feature_dct, probability))
        else:
            prediction = self.predict_one_example(features, probability)

        return prediction

    def predict(self, videos, probability=False):

        # TODO: extract_features_from_all_videos should be able to deal with features only (no labels)
        examples_features = extract_features_from_all_videos(videos,
                                                             self.feature_extractors,
                                                             classifier='logistic')
        predictions = self.predict_with_features(examples_features, probability)

        return predictions

    def save(self):

        model_name = 'model-'+self.name
        with open(model_name, "wb") as f:
            dill.dump(self, f)

        print("%s saved" % model_name)


    @staticmethod
    def logistic_fn(xw):

        return 1.0 / (1.0 + math.exp(-xw))

    @staticmethod
    def dotProduct(self, d1, d2):
        """
        @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
        @param dict d2: same as d1
        @return float: the dot product between d1 and d2
        """
        if len(d1) < len(d2):
            return self.dotProduct(d2, d1)
        else:
            return sum(d1.get(f, 0) * v for f, v in d2.items())

    def verbosePredict(self, phi, out, y=None):

        pred = self.predict_with_features(phi)

        if y:
            print >> out, 'Truth: %s, Prediction: %s [%s]' % (y, pred, 'CORRECT' if y == pred else 'WRONG')
        else:
            print >> out, 'Prediction:', pred
        for f, v in sorted(phi.items(), key=lambda (f, v): -v * self.weights.get(f, 0)):
            w = self.weights.get(f, 0)
            print >> out, "%-30s%s * %s = %s" % (f, v, w, v * w)

        return

    def output_error_analysis(self, examples):

        out = open('error-analysis-'+self.name, 'w')

        for example in examples:
            if type(example) is tuple:
                x, y = example
                print >> out, '===', x['url']
                x, y = extract_features_from_all_videos([example], self.feature_extractors, classifier='logistic')[0]
            elif type(example) is dict:
                x = example
                y = None
                print >> out, '===', x['url']
                x = extract_features_from_all_videos([example], self.feature_extractors, classifier='logistic')[0]
            else:
                raise ValueError("Examples have to come as a list of examples or as a dict of one example")

            self.verbosePredict(x, out, y)
        out.close()


def create_sparse_matrix(train_examples):

    y = []
    rows = []
    cols = []
    data = []
    feature_to_col = {}
    row = 0
    col = 0

    for feature_dct, label in train_examples:
        y.append(label)
        for feature, cnt in feature_dct.iteritems():

            rows.append(row)

            if feature not in feature_to_col:
                feature_to_col[feature] = col
                col += 1
            cols.append(feature_to_col[feature])

            data.append(cnt)

        row += 1
    #pdb.set_trace()

    return coo_matrix((data, (rows, cols)), shape=(row, col)), y


def learn_SVM_sklearn(train_examples, validation_examples, feature_extractor,
                            regularization=None, num_epochs=20, alpha=0.03, lambda_=0.01):
    pdb.set_trace()
    train_examples = extract_features_from_all_videos(train_examples, feature_extractor)
    validation_examples = extract_features_from_all_videos(validation_examples, feature_extractor)

    X, y = create_sparse_matrix(train_examples + validation_examples)
    model = SVC(kernel='linear')
    train_sizes, train_scores, valid_scores = learning_curve(model, X, y)

    print(train_scores)
    print(valid_scores)
    print(train_sizes)
    pdb.set_trace()

    return model.coef_


def learn_logistic_sklearn(train_examples, validation_examples, feature_extractor,
                            regularization=None, num_epochs=20, alpha=0.03, lambda_=0.01):

    train_examples = extract_features_from_all_videos(train_examples, feature_extractor, classifier='logistic')
    validation_examples = extract_features_from_all_videos(validation_examples, feature_extractor, classifier='logistic')

    X, y = create_sparse_matrix(train_examples+validation_examples)
    model = LogisticRegression(C=0.05)
    model.fit(X, y)

    X, y = create_sparse_matrix(validation_examples)
    pdb.set_trace()

    evaluate_predictor(zip(X, y), model.predict)

    #train_sizes, train_scores, valid_scores = learning_curve(model, X, y)

    #print(train_scores)
    #print(valid_scores)
    #print(train_sizes)


    return model.coef_


def evaluate_predictor(examples, predictor):
    '''
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
    of misclassiied examples.
    '''

    error = 0
    for x, y in examples:
        if predictor(x) != y:
            error += 1
    return 1.0 * error / len(examples)


# def dotProduct(d1, d2):
#     """
#     @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
#     @param dict d2: same as d1
#     @return float: the dot product between d1 and d2
#     """
#     if len(d1) < len(d2):
#         return dotProduct(d2, d1)
#     else:
#         return sum(d1.get(f, 0) * v for f, v in d2.items())
#
#
# def verbosePredict(phi, y, weights, out):
#     yy = 1 if dotProduct(phi, weights) > 0 else -1
#     if y:
#         print >>out, 'Truth: %s, Prediction: %s [%s]' % (y, yy, 'CORRECT' if y == yy else 'WRONG')
#     else:
#         print >>out, 'Prediction:', yy
#     for f, v in sorted(phi.items(), key=lambda (f, v) : -v * weights.get(f, 0)):
#         w = weights.get(f, 0)
#         print >> out, "%-30s%s * %s = %s" % (f, v, w, v * w)
#
#     return yy
#
#
# def outputErrorAnalysis(examples, featureExtractor, weights, path):
#     # TODO: this only takes one feature extractor
#     out = open(path, 'w')
#     for x, y in examples:
#         print >>out, '===', x['url']
#         verbosePredict(featureExtractor(x), y, weights, out)
#     out.close()


def outputWeights(weights, path):
    print "%d weights" % len(weights)
    out = open(path, 'w')
    for f, v in sorted(weights.items(), key=lambda (f, v) : -v):
        print >>out, '\t'.join([f, str(v)])

    out.close()


def load_weights(path):
    weights = Counter()
    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        feature, weight = line.split('\t')
        weights[feature] = float(weight)

    return weights


def train(data):

    examples = []
    for key, video in data['comedy'].iteritems():
        video['url'] = 'https://www.youtube.com/watch?v='+key
        examples.append((video, 1))

    for key, video in data['no_comedy'].iteritems():
        video['url'] = 'https://www.youtube.com/watch?v='+key
        examples.append((video, 0))

    # with open('oversample', 'r') as f:
    #     lines = f.readlines()
    #
    # for line in lines:
    #     key, label = line.split()
    #     video = videos_data['music'][key]
    #     video['url'] = 'https://www.youtube.com/watch?v='+key
    #     for i in range(10):
    #         examples.append((video, int(label)))

    random.shuffle(examples)

    train_examples = examples[:int(len(examples)*0.7)]
    validation_examples1 = examples[int(len(examples)*0.7):]
    #validation_examples2 = examples[int(len(examples) * 0.8):]


    # word features
    model = Logistic_Regression(name='words', feature_extractors=[word_features_from_video(kgram_size=1),
                                                                         tag_features_from_video], regularization=None,
                                num_epochs=200, alpha=7e-5, lambda_=0.001
                                )

    model.fit(train_examples, validation_examples1)
    model.output_error_analysis(random.sample(validation_examples1, 20))
    model.save()

    # with open('model-words.pkl', "wb") as f:
    #     pickle.dump(model, f)

    #outputErrorAnalysis(random.sample(validation_examples1, 20), word_features_from_video(kgram_size=1),
    #                    model.weights, 'error-analysis-text')
    outputWeights(model.weights, 'weights-text')

    # # tag features
    # weights = learn_logistic_regression(train_examples, validation_examples1, tag_features_from_video,
    #                                     regularization=None, num_epochs=200, alpha=1e-4, lambda_=0.001)
    # outputErrorAnalysis(random.sample(validation_examples1, 20), word_features_from_video(kgram_size=1),
    #                     weights, 'error-analysis-tags')
    # outputWeights(weights, 'weights-tags')
    #
    #
    # # statistics
    # weights = learn_logistic_regression(train_examples, validation_examples1, statistics_features_from_video,
    #                                     regularization=None, num_epochs=200, alpha=1e-4, lambda_=0.001)
    # outputErrorAnalysis(random.sample(validation_examples1, 20), word_features_from_video(kgram_size=1),
    #                     weights, 'error-analysis-statistics')
    # outputWeights(weights, 'weights-statistics')
    #
    # # Ensamble using Generalized additive models (GAM)
    # learn_ensamble(train_examples=validation_examples1, validation_examples=validation_examples2, models=[], feature_extractors=[])
    #


def score_music_videos(data, model):

    examples = []
    for key, video in data['music'].iteritems():
        video['url'] = 'https://www.youtube.com/watch?v=' + key
        examples.append(video)

    predictions = model.predict(examples, probability=True)

    sorted_url = []
    predictions_dct = {}
    for i, (example, prediction) in enumerate([(e, p) for p, e in sorted(zip(predictions, examples), reverse=True)]):
        predictions_dct[example['url']] = prediction
        sorted_url.append(example['url'])

    for i, url in enumerate(sorted_url):
        print url
        if i%10 == 0:
            pdb.set_trace()

    #model.output_error_analysis(to)
    pdb.set_trace()


def load_model(path):
    with open(path, 'rb') as f:
        model = dill.load(f)
    return model


def load_data(path='videos_data'):
    with open('videos_data', 'rb') as f:
        data = dill.load(f)
    return data


def main():
    data = load_data()
    train(data)
    model = load_model('model-words')
    score_music_videos(data, model)


if __name__=='__main__':
    main()
