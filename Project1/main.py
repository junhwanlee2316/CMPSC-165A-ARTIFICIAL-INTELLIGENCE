from classifier import load_data,tokenize, feature_extractor, data_processor
from classifier import classifier_agent

import numpy as np


def main():
    print("Creating a classifier agent:")

    with open('data/vocab.txt') as file:
        reading = file.readlines()
        vocab_list = [item.strip() for item in reading]
        vocab_dict = {item: i for i, item in enumerate(vocab_list)}

    print("Loading and processing data ...")

    sentences_pos = load_data("data/training_pos.txt")
    sentences_neg = load_data("data/training_neg.txt")

    train_sentences = sentences_pos + sentences_neg

    train_labels = [1 for i in range(len(sentences_pos))] + [0 for i in range(len(sentences_neg))]

    sentences_pos = load_data("data/test_pos_public.txt")
    sentences_neg = load_data("data/test_neg_public.txt")
    test_sentences = sentences_pos + sentences_neg
    test_labels = [1 for i in range(len(sentences_pos))] + [0 for i in range(len(sentences_neg))]


    feat_map = feature_extractor(vocab_list, tokenize)
    # You many replace this with a different feature extractor

    # feat_map = tfidf_extractor(vocab_list, tokenize, word_freq)

    # Preprocess the training data into features

    text2feat = data_processor(feat_map)
    Xtrain, ytrain = text2feat.process_data_and_save_as_file(train_sentences, train_labels, "train_data.npy")

    # Load the saved Xtrain

    #Xtrain = text2feat.load_data_from_file("train_data.npy")



    # train with GD
    niter = 1000
    print("Training using GD for ", niter, "iterations.")
    d = len(vocab_list)
    params = np.array([0.0 for i in range(d)])
    classifier1 = classifier_agent(feat_map,params)
    classifier1.train_gd(Xtrain,ytrain,niter,0.01,RAW_TEXT=False)




    # train with SGD
    nepoch = 10
    print("Training using SGD for ", nepoch, "data passes.")
    d = len(vocab_list)
    params = np.array([0.0 for i in range(d)])
    classifier2 = classifier_agent(feat_map, params)
    classifier2.train_sgd(Xtrain, ytrain, nepoch, 0.001,RAW_TEXT=False)


    err1 = classifier1.eval_model(test_sentences,test_labels)
    err2 = classifier2.eval_model(test_sentences,test_labels)

    print('GD: test err = ', err1,
          'SGD: test err = ', err2)


if __name__ == "__main__":
    main()
