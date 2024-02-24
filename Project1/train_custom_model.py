from classifier import load_data,tokenize, data_processor
from classifier import custom_feature_extractor, classifier_agent
from classifier import compute_twogram, custom_feature_extractor2


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
    # sentences_pos = load_data("../FullData/test_pos_private.txt")
    # sentences_neg = load_data("../FullData/test_neg_private.txt")
    test_sentences = sentences_pos + sentences_neg
    test_labels = [1 for i in range(len(sentences_pos))] + [0 for i in range(len(sentences_neg))]



    # TODO: ====================== Your code here ====================

    feat_map = custom_feature_extractor(vocab_list, tokenize)
    # You many replace this with a different feature extractor

    # sklearn_obj = compute_twogram(train_sentences)
    # feat_map = custom_feature_extractor2(vocab_list,tokenize,sklearn_obj)

    def MyFeatureMap(sentence):
        return feat_map(sentence)

    # TODO: ==========================================================



    # Preprocess the training data into features

    text2feat = data_processor(MyFeatureMap)
    Xtrain, ytrain = text2feat.process_data_and_save_as_file(train_sentences, train_labels,
                                            "custom_feat_train.npy")


    Xtest, ytest = text2feat.process_data_and_save_as_file(test_sentences,test_labels,
                                            "custom_feat_test.npy")
    # "custom_feat_test.npy" should be submitted to gradescope

    #Xtest = text2feat.load_data_from_file("custom_feat_test.npy")



    # train with SGD
    nepoch = 3
    print("Training using SGD for ", nepoch, "data passes.")
    d = len(vocab_list)

    params = np.array([0.0 for i in range(d)])
    custom_classifier = classifier_agent(MyFeatureMap, params)

    # TODO: ====================== Feel free to tweak how it is trained here====================
    custom_classifier.train_sgd(Xtrain, ytrain, nepoch,0.01, RAW_TEXT = False)

    ## Hint:
    # - if you use tf-idf then the appropriate scale of the learning rate might be a lot bigger
    #   due to normalization.

    #custom_classifier.train_sgd(train_sentences, train_labels, nepoch, 0.001)

    #niter = 2000
    #custom_classifier.train_gd(train_sentences, train_labels, niter, 100.0)

    err = custom_classifier.eval_model(test_sentences,test_labels)

    print("Test error =  ", err)

    custom_classifier.save_params_to_file('best_model.npy')


    # You will need to submit "best_model.npy" and "custom_feat_test.npy"



if __name__ == "__main__":
    main()
