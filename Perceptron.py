import sys
import time
import numpy as np
from Eval import Eval
from scipy.sparse import csr_matrix, hstack
from imdb import IMDBdata

class Perceptron:
    def __init__(self, X, Y, N_ITERATIONS):
        self.N_ITERATIONS = N_ITERATIONS
        #TODO: Initalize parameters
        self.vocab_len = X.shape[1]
        self.examples = X.shape[0]
        self.weight = np.zeros([1,X.shape[1]]) # +1 for the bias
        self.bias = 0
        self.for_avg_weight = 0
        self.for_avg_bias = 0
        self.penalty = 1
        self.Train(X,Y)

    def ComputeAverageParameters(self):
        #TODO: Compute average parameters (do this part last)
        self.weight = self.weight - (self.for_avg_weight / float(self.penalty))
        return

    def Train(self, X, Y):
        #TODO: Estimate perceptron parameters
        ite = self.N_ITERATIONS
        examples = self.examples
        activation = 0
        weight_transpose = np.zeros([X.shape[1],1])
        for i in range(ite) :
            print i
            converged = 1
            for j in range(examples):
                term = (X[j].dot(weight_transpose))
                activation = 0
                if(term > 0.0) :
                    activation = 1.0   
                elif term < 0.0 :
                    activation =-1.0                             
                if Y[j] != activation:
                    weight_transpose +=  (Y[j] * X[j].transpose())
                    self.for_avg_weight += Y[j] * self.penalty * X[j] 
                    converged = 0  
                self.penalty +=  1
            if converged == 1:
                break
        self.weight = weight_transpose.transpose()
        return

    def Predict(self, X):
        #TODO: Implement perceptron classification
        pred_labels = []
        examples = X.shape[0]
        weight_transpose = self.weight.transpose()
        for j in range(examples):
            if (X[j].dot(weight_transpose)) > 0 :
                pred_labels.append(1.0)
            else:
                pred_labels.append(-1.0)
        
        return pred_labels

    def Eval(self, X_test, Y_test):
        Y_pred = self.Predict(X_test)
        ev = Eval(Y_pred, Y_test)
        return ev.Accuracy()
    
    def Pos_Neg_words(self, vocab):
        weights = self.weight[:,1:]
        print weights.shape
        words = np.argsort(weights)
        le = words.shape[1]
        print "Most Positive Words"
        for i in range(20):            
            print vocab.GetWord(words.item(0, le - 1 - i)), "," , weights.item(0, words.item(0, le - 1 - i))
            
        print "Most Negative Words"
        for i in range(20):
            print vocab.GetWord(words.item(0, i)), "," , weights.item(0, words.item(0, i))
            
if __name__ == "__main__":
    
    print "Reading Training Data"
    train = IMDBdata("%s/train" % sys.argv[1])
    print "Reading Test Data"
    test  = IMDBdata("%s/test" % sys.argv[1], vocab=train.vocab)
    
    bias_train_val = np.ones((train.X.shape[0],))
    bias_test_val = np.ones((test.X.shape[0],))
    
    row_train = np.arange(train.X.shape[0])
    row_test = np.arange(test.X.shape[0])
    
    col_train = np.zeros((train.X.shape[0],))
    col_test = np.zeros((test.X.shape[0],))
    
    bias_train = csr_matrix((bias_train_val,(row_train, col_train)), shape=(train.X.shape[0],1))
    bias_test = csr_matrix((bias_test_val,(row_test, col_test)), shape=(test.X.shape[0],1))
    
    train.X = hstack([bias_train,train.X]).tocsr()
    test.X = hstack([bias_test,test.X]).tocsr()
    
    print "Training Perceptron"
    ptron = Perceptron(train.X, train.Y, int(sys.argv[2]))
    print "Evaluating Original Perceptron on test data"
    print ptron.Eval(test.X, test.Y)
    
    print "Computing Average Parameters of the Perceptron"
    ptron.ComputeAverageParameters()

    print "Evaluating Average Perceptron on test data"
    print ptron.Eval(test.X, test.Y)

    #TODO: Print out the 20 most positive and 20 most negative words
    ptron.Pos_Neg_words(train.vocab)