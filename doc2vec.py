#!/usr/bin/env python
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_svmlight_files
from scipy.sparse import hstack

from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import LabeledSentence


from KaggleWord2VecUtility import KaggleWord2VecUtility


def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0

    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    
    if nwords != 0:
        featureVec /= nwords
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0

    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")

    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, True))
    return clean_reviews 


def getFeatureVecs(reviews, model, num_features):
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    counter = -1
    
    for review in reviews:
        counter = counter + 1
        try:
            reviewFeatureVecs[counter] = np.array(model[review.labels[0]]).reshape((1, num_features))
        except:
            continue
    return reviewFeatureVecs


def getCleanLabeledReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, True))
    
    labelized = []
    for i, id_label in enumerate(reviews["id"]):
        labelized.append(LabeledSentence(clean_reviews[i], [id_label]))
    return labelized


if __name__ == '__main__':
    train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    test = pd.read_csv('testData.tsv', header=0, delimiter="\t", quoting=3 )
   
    print "Cleaning and parsing the data sets...\n"

    clean_train_reviews = []
    for review in train['review']:
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(review)))

    clean_test_reviews = []
    for review in test['review']:
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(review)))

    print "Creating the bag of words...\n"

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,3), sublinear_tf=True)
    
    X_train_bow = vectorizer.fit_transform(clean_train_reviews)
    X_test_bow = vectorizer.transform(clean_test_reviews)
    
       
    print "Cleaning and labeling the data sets...\n"
    
    train_reviews = getCleanLabeledReviews(train)
    test_reviews = getCleanLabeledReviews(test)

    n_dim = 5000
    
    print 'Loading doc2vec model..\n'
    
    model_dm_name = "%dfeatures_40minwords_10context_dm" % n_dim
          
    model_dm = Doc2Vec.load(model_dm_name)
        
    print "Creating the d2v vectors...\n"

    X_train_d2v_dm = getFeatureVecs(train_reviews, model_dm, n_dim)

    X_test_d2v_dm = getFeatureVecs(test_reviews, model_dm, n_dim)

    X_train_bdv = hstack([X_train_bow, X_train_d2v_dm])
    X_test_bdv = hstack([X_test_bow, X_test_d2v_dm])

    print "Checking the dimension of training vectors"
    
    print 'BoW', X_train_bow.shape
    print 'D2V', X_train_d2v_dm.shape
    print 'BoW-D2V', X_train_bdv.shape

    y_train = train['sentiment']
    
    
    print "Predicting with Bag-of-words model...\n" 
    
    clf = LogisticRegression(class_weight="auto")
    
    clf.fit(X_train_bow, y_train)
    y_prob_bow = clf.predict_proba(X_test_bow)

    print "Predicting with Bag-of-words model and Doc2Vec model...\n" 
	
    clf.fit(X_train_bdv, y_train)
    y_prob_bdv = clf.predict_proba(X_test_bdv)

    y_pred = 0.2*y_prob_bow + 0.8*y_prob_bdv

    output = pd.DataFrame(data={"id":test["id"], "sentiment":y_pred[:,1]})
    output.to_csv('doc2vec_bow.csv', index=False, quoting=3)
    
    print "Wrote results to doc2vec_bow.csv" 
