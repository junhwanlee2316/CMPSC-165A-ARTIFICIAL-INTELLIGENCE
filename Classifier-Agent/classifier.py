import scipy
from scipy import sparse
import numpy as np
from collections import Counter
import string

# utility functions we provides

def load_data(file_name):
    '''
    @input:
     file_name: a string. should be either "training.txt" or "texting.txt"
    @return:
     a list of sentences
    '''
    with open(file_name, "r", encoding='utf-8') as file:
        sentences = file.readlines()
    return sentences


def tokenize(sentence):
    # Convert a sentence into a list of words
    wordlist = sentence.translate(str.maketrans('', '', string.punctuation)).lower().strip().split(
        ' ')

    return [word.strip() for word in wordlist]


# Main "Feature Extractor" class:
# It takes the provided tokenizer and vocab as an input.

class feature_extractor:
    def __init__(self, vocab, tokenizer):
        self.tokenize = tokenizer
        self.vocab = vocab  # This is a list of words in vocabulary
        self.vocab_dict = {item: i for i, item in
                           enumerate(vocab)}  # This constructs a word 2 index dictionary
        self.d = len(vocab)

    def bag_of_word_feature(self, sentence):
        '''
        Bag of word feature extactor
        :param sentence: A text string representing one "movie review"
        :return: The feature vector in the form of a "sparse.csc_array" with shape = (d,1)
        '''

        # TODO ======================== YOUR CODE HERE =====================================
        # Hint 1:  there are multiple ways of instantiating a sparse csc matrix.
        #  Do NOT construct a dense numpy.array then convert to sparse.csc_array. That will defeat its purpose.

        # Hint 2:  There might be words from the input sentence not in the vocab_dict when we try to use this.

        # Hint 3:  Python's standard library: Collections.Counter might be useful
                
        # TODO =============================================================================

        # Solution:
        
        

        word_count = Counter(self.tokenize(sentence))
        indices, values = [],[]
        
    
        for word, count in word_count.items():
            if word in self.vocab_dict:
                indices.append(self.vocab_dict[word])
                values.append(count)
        
        indices,values = np.array(indices), np.array(values)
        x = sparse.csc_array((values, (indices, np.zeros_like(indices))), shape = (self.d, 1))
        return x

                
        

    def __call__(self, sentence):
        # This function makes this any instance of this python class a callable object
        return self.bag_of_word_feature(sentence)


class data_processor:
    '''
    Please do NOT modify this class.
    This class basically takes any FeatureExtractor class, and provide utility functions
    1. to process data in batches
    2. to save them to npy files
    3. to load data from npy files.
    '''
    # This class
    def __init__(self,feat_map):
        self.feat_map = feat_map

    def batch_feat_map(self, sentences):
        '''
        This function processes data according to your feat_map. Please do not modify.

        :param sentences:  A single text string or a list of text string
        :return: the resulting feature matrix in sparse.csc_array of shape d by m
        '''
        if isinstance(sentences, list):
            X = scipy.sparse.hstack([self.feat_map(sentence) for sentence in sentences])
        else:
            X = self.feat_map(sentences)
        return X

    def load_data_from_file(self, filename):
        # The filename should be *.npy
        with open(filename, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            if data.shape == ():
                X = data[()]
            else:
                X = data
        return X

    def process_data_and_save_as_file(self,sentences,labels, filename):
        # The filename should be *.npy
        X = self.batch_feat_map(sentences)
        y = np.array(labels)
        with open(filename, 'wb') as f:
            np.save(f, X, allow_pickle=True)
        return X, y


class classifier_agent():
    def __init__(self, feat_map, params):
        '''
        This is a constructor of the 'classifier_agent' class. Please do not modify.

         - 'feat_map'  is a function that takes the raw data sentence and convert it
         into a data vector compatible with numpy.array

         Once you implement Bag Of Word and TF-IDF, you can pass an instantiated object
          of these class into this classifier agent

         - 'params' is an numpy array that describes the parameters of the model.
          In a linear classifer, this is the coefficient vector. This can be a zero-initialization
          if the classifier is not trained, but you need to make sure that the dimension is correct.
        '''
        self.feat_map = feat_map
        self.data2feat = data_processor(feat_map)
        self.batch_feat_map = self.data2feat.batch_feat_map

        self.params = np.array(params)

    def score_function(self, X):
        '''
        This function computes the score function of the classifier.
        Note that the score function is linear in X
        :param X: A scipy.sparse.csc_array of size d by m, each column denotes one feature vector
        :return: A numpy.array of length m with the score computed for each data point
        '''

        (d,m) = X.shape
        d1= self.params.shape[0]
        if d != d1:
            self.params = np.array([0.0 for i in range(d)])

        # TODO ======================== YOUR CODE HERE =====================================
        s = X.T.dot(self.params)  # this is the desired type and shape for the output
        
        # the score function is NOT a soft probabilistic prediction.
        # It is the score the classifier used to compare different classes,
        # e.g., for linear classifier it is the weighted linear combination of the features
        #       in decision tree classifiers, it is the voting score at each leaf node.
        # TODO =============================================================================
        return s



    def predict(self, X, RAW_TEXT=False, RETURN_SCORE=False):
        '''
        This function makes a binary prediction or a numerical score
        :param X: d by m sparse (csc_array) matrix
        :param RAW_TEXT: if True, then X is a list of text string
        :param RETURN_SCORE: If True, then return the score directly
        :return:
        '''
        if RAW_TEXT:
            X = self.batch_feat_map(X)

        # TODO ======================== YOUR CODE HERE =====================================
        # This should be a simple but useful function.
        # Tip:   Read the required format of the predictions.

        if RETURN_SCORE:
            return self.score_function(X)
        else:
            score = self.score_function(X)
            preds = np.where(score > 0, 1, 0)
        
        # TODO =============================================================================
        
        return preds




    def error(self, X, y, RAW_TEXT=False):
        '''
        :param X: d by m sparse (csc_array) matrix
        :param y: m dimensional vector (numpy.array) of true labels
        :param RAW_TEXT: if True, then X is a list of text string,
                        and y is a list of true labels
        :return: The average error rate
        '''
        if RAW_TEXT:
            X = self.batch_feat_map(X)
            y = np.array(y)

        # TODO ======================== YOUR CODE HERE =====================================
        # The function should work for any integer m > 0.
        # You may wish to use self.predict
        preds = self.predict(X)
        err =  np.mean(preds != y)
        # TODO =============================================================================

        return err

    def loss_function(self, X, y):
        '''
        This function implements the logistic loss at the current self.params

        :param X: d by m sparse (csc_array) matrix
        :param y: m dimensional vector (numpy.array) of true labels
        :return:  a scalar, which denotes the mean of the loss functions on the m data points.

        '''

        # TODO ======================== YOUR CODE HERE =====================================
        # The function should work for any integer m > 0.
        # You may first call score_function
        # Compute logistic function probabilities
        
        scores = self.score_function(X)
        log_sum_exp = np.logaddexp(0, scores)
        log_p = scores - log_sum_exp
        
        loss = - y * log_p - (1 - y) * (-log_sum_exp)
        # TODO =============================================================================

        #return loss
        return np.mean(loss)

    def gradient(self, X, y):
        '''
        It returns the gradient of the (average) loss function at the current params.
        :param X: d by m sparse (csc_array) matrix
        :param y: m dimensional vector (numpy.array) of true labels
        :return: Return an nd.array of size the same as self.params
        '''

        # TODO ======================== YOUR CODE HERE =====================================
        # Hint 1:  Use the score_function first
        # Hint 2:  vectorized operations will be orders of magnitudely faster than a for loop
        # Hint 3:  don't make X a dense matrix
        scores = self.score_function(X)

        # Compute probabilities
        probs = np.exp(scores) / (1 + np.exp(scores))

        # Compute gradient
        grad = X.dot(probs - y)/X.shape[1]
        # TODO =============================================================================
        return grad




    def train_gd(self, train_sentences, train_labels, niter, lr=0.01, RAW_TEXT=True):
        '''
        The function should updates the parameters of the model for niter iterations using Gradient Descent
        It returns the sequence of loss functions and the sequence of training error for each iteration.

        By default the function takes raw text. But it also takes already pre-processed features,
        if RAW_TEXT is set to False.

        :param train_sentences: Training data, a list of text strings;
            when "RAW_TEXT" is set to False, this input is a d by n numpy.array or scipy.csc_array
        :param train_labels: Training data, a list of labels 0 or 1
            when "RAW_TEXT" is set to False, this input is a n dimensional numpy.array
        :param niter: number of iterations to train with Gradient Descent
        :param lr: Choice of learning rate (default to 0.01, but feel free to tweak it)
        :return: A list of loss values, and a list of training errors.
                (Both of them has length niter + 1)
        '''
        if RAW_TEXT:
            # the input is raw text
            Xtrain = self.batch_feat_map(train_sentences)
            ytrain = np.array(train_labels)
        else:
            # the input is the extracted feature vector
            Xtrain = train_sentences
            ytrain = train_labels

        train_losses = [self.loss_function(Xtrain, ytrain)]
        train_errors = [self.error(Xtrain, ytrain)]

        # Solution:
        for i in range(niter):
            # TODO ======================== YOUR CODE HERE =====================================
            # You need to iteratively update self.params
            grad = self.gradient(Xtrain, ytrain)
            self.params -= lr * grad

            # TODO =============================================================================
            train_losses.append(self.loss_function(Xtrain, ytrain))
            train_errors.append(self.error(Xtrain, ytrain))

            if i%100 == 0:
                print('iter =',i,'loss = ', train_losses[-1],
                  'error = ', train_errors[-1])

        return train_losses, train_errors



    def train_sgd(self, train_sentences, train_labels, nepoch, lr=0.001, RAW_TEXT=True):
        '''
        The function should updates the parameters of the model for using Stochastic Gradient Descent.
        (random sample in every iteration, without minibatches,
        pls follow the algorithm from the lecture which picks one data point at random).

        By default the function takes raw text. But it also takes already pre-processed features,
        if RAW_TEXT is set to False.


        :param train_sentences: Training data, a list of text strings;
            when "RAW_TEXT" is set to False, this input is a d by n numpy.array or scipy.csc_array
        :param train_labels: Training data, a list of labels 0 or 1
            when "RAW_TEXT" is set to False, this input is a n dimensional numpy.array
        :param nepoch: Number of effective data passes.  One data pass is the same as n iterations
        :param lr: Choice of learning rate (default to 0.001, but feel free to tweak it)
        :return: A list of loss values and a list of training errors.
                (initial loss / error plus  loss / error after every epoch, thus length epoch +1)
        '''


        if RAW_TEXT:
            # the input is raw text
            Xtrain = self.batch_feat_map(train_sentences)
            ytrain = np.array(train_labels)
        else:
            # the input is the extracted feature vector
            Xtrain = train_sentences
            ytrain = train_labels

        train_losses = [self.loss_function(Xtrain, ytrain)]
        train_errors = [self.error(Xtrain, ytrain)]


        # First construct the dataset
        # then train the model using SGD
        # Solution
        sampler = 1/len(ytrain)
        niter = int(nepoch / sampler)

        #params = sparse.csr_array(self.params)

        for i in range(nepoch):
            for j in range(len(ytrain)):

            # TODO ======================== YOUR CODE HERE =====================================
            # You need to iteratively update self.params
            # You should use the following for selecting the index of one random data point.

                idx = np.random.choice(len(ytrain), 1)
                grad = self.gradient(Xtrain[:, idx[0]: idx[0]+1], ytrain[idx[0]:idx[0]+1])
                self.params -= lr * grad
            # TODO =============================================================================
            # logging
            train_losses.append(self.loss_function(Xtrain, ytrain))
            train_errors.append(self.error(Xtrain, ytrain))

            print('epoch =',i,'iter=',i*len(ytrain)+j+1,'loss = ', train_losses[-1],
                  'error = ', train_errors[-1])


        return train_losses, train_errors


    def eval_model(self, test_sentences, test_labels, RAW_TEXT=True):
        '''
        This function evaluates the classifier agent via new labeled examples.
        Do not edit please.
        :param test_sentences: Test data, a list of text strings;
            when "RAW_TEXT" is set to False, this input is a d by n numpy.array or scipy.csc_array
        :param test_labels: Test data, a list of labels 0 or 1
            when "RAW_TEXT" is set to False, this input is a n dimensional numpy.array
        :return: error rate on the input dataset
        '''

        if RAW_TEXT:
            # the input is raw text
            X = self.batch_feat_map(test_sentences)
            y = np.array(test_labels)
        else:
            # the input is the extracted feature vector
            X = test_sentences
            y = test_labels

        return self.error(X, y)

    def save_params_to_file(self, filename):
        # The filename should be *.npy
        with open(filename, 'wb') as f:
            np.save(f, self.params)

    def load_params_from_file(self, filename):
        with open(filename, 'rb') as f:
            self.params = np.load(f)





class custom_feature_extractor(feature_extractor):
    '''
    This is a template for implementing more advanced feature extractor
    '''
    def __init__(self, vocab, tokenizer, other_inputs=None):
        super().__init__(vocab, tokenizer)
        # TODO ======================== YOUR CODE HERE =====================================
        # Adding external inputs that need to be saved.
        # TODO =============================================================================

    def feature_map(self,sentence):
        # -------- Your implementation of the advanced feature ---------------
        # TODO ======================== YOUR CODE HERE =====================================
        x = self.bag_of_word_feature(sentence)
        # Implementing the advanced feature.
        # TODO =============================================================================
        return x

    def __call__(self, sentence):
        # If you don't edit anything you will use the standard bag of words feature
        return self.feature_map(sentence)


## You are free to do anything you want for your feature engineering task
# And you can use any external package for the task
## Some ideas (read about them!):
#  1. n-grams   2. Removing "Stop words"  3.  tf-idf  4. word2vec
# 5. pre-trained embedding using e.g., Bert.

# An example on how to construct an off-the-shelf 2-gram feature extractor is provided below.


# Some of these may need some data-structure to be set up and stored, rather than computing from
# scratch each time, e.g., the idf part of tf-idf should be computed once and stored as an attribute
# of the feature extractor class

def compute_twogram(train_sentences):

    from sklearn.feature_extraction.text import CountVectorizer

    gram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    gram_vectorizer.fit(train_sentences)
    return gram_vectorizer


class custom_feature_extractor2(feature_extractor):
    '''
    This is a template for implementing more advanced feature extractor

    You my call it by, e.g.,

    twogram = compute_twogram(train_sentences)
    custom_feat_map = custom_feature_extractor2(vocab,tokenizer,twogram)

    # Notice that this might not even use our "vocab" or "tokenizer" at all and it is okay!

    '''
    def __init__(self, vocab, tokenizer, sklearn_transform):
        super().__init__(vocab, tokenizer)
        # TODO ======================== YOUR CODE HERE =====================================
        # Adding external inputs that need to be saved.
        self.ngram=sklearn_transform.transform
        # TODO =============================================================================

    def feature_map(self,sentence):
        # -------- Your implementation of the advanced feature ---------------
        # TODO ======================== YOUR CODE HERE =====================================
        #x = self.bag_of_word_feature(sentence)
        # Implementing the advanced feature.
        x = self.ngram([sentence]).T
        # TODO =============================================================================
        return x

    def __call__(self, sentence):
        # If you don't edit anything you will use the standard bag of words feature
        return self.feature_map(sentence)




class custom_feature_extractor2(feature_extractor):
    '''
    This is a template for implementing more advanced feature extractor
    '''
    def __init__(self, vocab, tokenizer, sklearn_transform):
        super().__init__(vocab, tokenizer)
        # TODO ======================== YOUR CODE HERE =====================================
        # Adding external inputs that need to be saved.
        self.ngram=sklearn_transform.transform
        # TODO =============================================================================

    def feature_map(self,sentence):
        # -------- Your implementation of the advanced feature ---------------
        # TODO ======================== YOUR CODE HERE =====================================
        #x = self.bag_of_word_feature(sentence)
        # Implementing the advanced feature.
        x = self.ngram([sentence]).T
        # TODO =============================================================================
        return x

    def __call__(self, sentence):
        # If you don't edit anything you will use the standard bag of words feature
        return self.feature_map(sentence)



