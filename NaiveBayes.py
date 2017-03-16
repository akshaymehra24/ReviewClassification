import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import time
from imdb import IMDBdata

class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.data = data # training data
        #TODO: Initalize parameters
        self.vocab_len = data.X.shape[1]
        self.count_positive = np.zeros([1,data.X.shape[1]])
        self.count_negative = np.zeros([1,data.X.shape[1]])
        self.num_positive_reviews = 0
        self.num_negative_reviews = 0
        self.total_positive_words = 0
        self.total_negative_words = 0
        self.P_positive = 0.0
        self.P_negative = 0.0
        self.deno_pos = 0.0
        self.deno_neg = 0.0
        self.Train(data.X,data.Y)

    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        positive_indices = np.argwhere(Y == 1.0).flatten()
        negative_indices = np.argwhere(Y == -1.0).flatten()
        
        self.num_positive_reviews = len(positive_indices)
        self.num_negative_reviews = len(negative_indices)
        
        self.count_positive = csr_matrix.sum(X[np.ix_(positive_indices)], axis=0)
        self.count_negative = csr_matrix.sum(X[np.ix_(negative_indices)], axis=0)
        
        self.total_positive_words = csr_matrix.sum(X[np.ix_(positive_indices)])
        self.total_negative_words = csr_matrix.sum(X[np.ix_(negative_indices)])
        
        self.deno_pos = float(self.total_positive_words + self.ALPHA * X.shape[1])
        self.deno_neg = float(self.total_negative_words + self.ALPHA * X.shape[1])
        
        self.count_positive = (self.count_positive + self.ALPHA) 
        self.count_negative = (self.count_negative + self.ALPHA) 
        
        return

    def PredictLabel(self, X):
        #TODO: Implement Naive Bayes Classification
        self.P_positive = log(float(self.num_positive_reviews)) - log(float(self.num_positive_reviews + self.num_negative_reviews))
        self.P_negative = log(float(self.num_negative_reviews)) - log(float(self.num_positive_reviews + self.num_negative_reviews))
        pred_labels = []
        
        sh = X.shape[0]
        for i in range(sh):
            z = X[i].nonzero()
            sum_positive = self.P_positive
            sum_negative = self.P_negative
            for j in range(len(z[0])):
                row_index = i
                col_index = z[1][j]
                times = X[row_index, col_index]
        
                P_pos = log(self.count_positive[0, col_index]) - log(self.deno_pos)
                sum_positive = sum_positive + times * P_pos
                
                P_neg = log(self.count_negative[0, col_index]) - log(self.deno_neg)
                sum_negative = sum_negative + times * P_neg
            
            if sum_positive > sum_negative:
                pred_labels.append(1.0)
            else:
                pred_labels.append(-1.0)
        
        return pred_labels
        #return 1

    def LogSum(self, logx, logy):   
        # TO Do: Return log(x+y), avoiding numerical underflow/overflow.
        m = max(logx, logy)        
        return m + log(exp(logx - m) + exp(logy - m))

    def PredictProb(self, test, indexes):
    
        for i in indexes:
            # TO DO: Predict the probability of the i_th review in test being positive review
            # TO DO: Use the LogSum function to avoid underflow/overflow
            predicted_label = 0
            z = test.X[i].nonzero()
            sum_positive = self.P_positive
            sum_negative = self.P_negative
            for j in range(len(z[0])):
                row_index = i
                col_index = z[1][j]
                times = test.X[row_index, col_index]
                
                P_pos = log((self.count_positive[0, col_index]))
                sum_positive = sum_positive + times * P_pos
                    
                P_neg = log((self.count_negative[0, col_index]))
                sum_negative = sum_negative + times * P_neg
            
            predicted_prob_positive = exp(sum_positive - self.LogSum(sum_positive, sum_negative))
            predicted_prob_negative = exp(sum_negative - self.LogSum(sum_positive, sum_negative))
            
            if sum_positive > sum_negative:
                predicted_label = 1.0
            else:
                predicted_label = -1.0
            
            #print test.Y[i], test.X_reviews[i]
            # TO DO: Comment the line above, and uncomment the line below
            print test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative, test.X_reviews[i]


    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X)
        ev = Eval(Y_pred, test.Y)
        return ev.Accuracy()


if __name__ == "__main__":
    
    print "Reading Training Data"
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print "Reading Test Data"
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)    
    print "Computing Parameters"
    nb = NaiveBayes(traindata, float(sys.argv[2]))
    print "Evaluating"
    print "Test Accuracy: ", nb.Eval(testdata)
    print nb.PredictProb(testdata, range(10))  # predict the probabilities of reviews being positive (first 10 reviews in the test set)