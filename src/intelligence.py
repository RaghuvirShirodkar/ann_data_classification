import numpy as np

from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

from datetime import datetime

class Intelligence(object):

    def __init__(self, negative_, positive_, no_test_points):
        
        np.random.shuffle(negative_)
        np.random.shuffle(positive_)
        self.train_data = np.vstack((negative_[:no_test_points, :-1], positive_[:no_test_points, :-1]))
        self.train_labels = np.hstack((negative_[:no_test_points, -1], positive_[:no_test_points, -1]))
        self.test_data = np.vstack((negative_[no_test_points:, :-1], positive_[no_test_points:, :-1]))
        self.test_labels = np.hstack((negative_[no_test_points:, -1], positive_[no_test_points:, -1]))

    def svm_(self, kernel):
        '''
        Implement SVM with user defined kernel.
        :param:
            kernel: SVM kernel to be used
        :returns:
            None
        '''
        start_ = datetime.timestamp(datetime.now())
        svm_classifier = svm.SVC(C=500, kernel=kernel)
        svm_classifier.fit(self.train_data, self.train_labels)
        stop_ = datetime.timestamp(datetime.now())
        self.time_ = stop_ - start_
        self.__print_results(model=svm_classifier, desc='SVM classifier with '+kernel+' kernel')
        del svm_classifier

    def mlp_(self):
        '''
        Implement MLP with 1 input layer, 3 hidden
        layers and 1 output layer.
        :param: 
            None
        :returns:
            None
        '''
        start_ = datetime.timestamp(datetime.now())
        input_ = Input(shape=(self.train_data.shape[1],))
        hidden_1 = Dense(6, activation='sigmoid')(input_)
        hidden_2 = Dense(6, activation='sigmoid')(hidden_1)
        hidden_3 = Dense(6, activation='sigmoid')(hidden_2)
        output_ = Dense(1, activation='sigmoid')(hidden_3)
        mlp_model = Model(input_, output_)

        mlp_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01))

        weights = compute_class_weight('balanced', np.array([0,1]), self.train_labels)
        mlp_model.fit(self.train_data, self.train_labels, 
        batch_size=10, epochs=100, class_weight={0:weights[0],1:weights[1]})
        predicted_labels = mlp_model.predict(self.test_data)
        predicted_labels = np.array([i > 0.5 for i in predicted_labels])
        stop_ = datetime.timestamp(datetime.now())
        self.time_ = stop_ - start_
        self.__print_results(model=predicted_labels, desc='MLP using 1-3-1 network')
        del mlp_model

    def __print_results(self, model, desc):
        '''
        Print the results to verify network.
        :param:
            model: Model used for classification
            desc: Description
        :returns:
            None
        '''
        model = model.predict(self.test_data) if desc[:3]=='SVM' else model
        print (desc)
        print ('-'*50)
        print ('='*5+' Train/Test Split '+'='*5+'\n {:.2f}/{:.2f} %'\
               .format(len(self.train_data)/(len(self.train_data)+len(self.test_data))*100, \
                       len(self.test_data)/(len(self.train_data)+len(self.test_data))*100))
        print ('='*5+' Confusion Matrix '+'='*5+'\n', confusion_matrix(self.test_labels, model, [1,0]))
        print ('='*5+' Precision '+'='*5+'\n {:.2f}'.format(precision_score(self.test_labels, model, [1,0])*100))
        print ('='*5+' Recall '+'='*5+'\n {:.2f}'.format(recall_score(self.test_labels, model, [1,0])))
        print ('='*5+' Execution Time '+'='*5+'\n {:.5f} sec'.format(self.time_))
        print ('-'*50+'\n')
