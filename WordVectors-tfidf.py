
# coding: utf-8

# # Deep Learning goes to movies
# # Word2Vec algorithm
# 

# In[ ]:

#Python implementation of word2vec from the gensim package
#In order to train your model in a reasonable amount of time, you will need to install cython 
#Word2Vec does not need labels in order to create meaningful representations. 
#This is useful, since most data in the real world is unlabeled


# In[1]:

import pandas as pd
from sklearn.linear_model import SGDClassifier

# Read data from files 
train = pd.read_csv( "labeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )

# Verify the number of reviews that were read (100,000 in total)
print "Read %d labeled train reviews, %d labeled test reviews, "  "and %d unlabeled reviews\n" % (train["review"].size,  
 test["review"].size, unlabeled_train["review"].size )


# In[2]:

# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review,"lxml").get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)


# In[4]:

#First, to train Word2Vec it is better not to remove stop words 
#because the algorithm relies on the broader context of the sentence 
#in order to produce high-quality word vectors.
# Download the punkt tokenizer for sentence splitting
import nltk.data
#nltk.download()   

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

#Load the model (this will take some time)
from gensim.models import Word2Vec

model = Word2Vec.load("5000features_40minwords_10context")


# In[16]:

model.syn0.shape


# In[17]:

model["flower"]


# In[19]:

import numpy as np  # Make sure that numpy is imported

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model,            num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


# In[20]:

# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.

clean_train_reviews = []
for review in train['review']:
    clean_train_reviews.append( " ".join( review_to_wordlist( review )))

unlabeled_clean_train_reviews = []
for review in unlabeled_train['review']:
    unlabeled_clean_train_reviews.append( " ".join( review_to_wordlist( review )))

print "Parsing test reviews..."

clean_test_reviews = []
for review in test['review']:
    clean_test_reviews.append( " ".join( review_to_wordlist( review )))

# In[21]:
num_features = 5000
#trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )
#testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

#from sklearn.preprocessing import Imputer
#testDataVecs =  Imputer().fit_transform(testDataVecs)
#trainDataVecs = Imputer().fit_transform(trainDataVecs)

# In[ ]:

# Fit a random forest to the training data, using 100 trees
#from sklearn.ensemble import RandomForestClassifier
#forest = RandomForestClassifier( n_estimators = 100 )

#print "Fitting a random forest to labeled training data..."
#forest = forest.fit( trainDataVecs, train["sentiment"] )

# Test & extract results 
#result = forest.predict_proba( testDataVecs )

# Write the test results 
#output = pd.DataFrame( data={"id":test["id"], "sentiment":result[:,1]} )
#output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )

# In[ ]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
import time
print "vectorizing..."
start = time.time() # Start time
vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features = 5000, ngram_range = ( 1, 3 ),
                              sublinear_tf = True )
# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print "Time taken for TF-IDF Vectorizer: ", elapsed, "seconds."

vectorizer = vectorizer.fit(clean_train_reviews + unlabeled_clean_train_reviews)
train_data_features = vectorizer.transform( clean_train_reviews )
test_data_features = vectorizer.transform( clean_test_reviews )


# In[23]:

model2 = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
model2.fit( train_data_features, train["sentiment"] )


# In[ ]:

p2 = model2.predict_proba( test_data_features )[:,1]

print "Writing results..."

output = pd.DataFrame( data = { "id": test["id"], "sentiment": p2 } )
output.to_csv( "Word2Vec_tfidf.csv", index = False, quoting = 3 )


# In[ ]:



