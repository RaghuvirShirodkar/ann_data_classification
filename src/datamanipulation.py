import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

class DataManipulation(object):
    
    def __init__(self, filename, sep_=';', dec_=','):

        self.data = pd.read_csv(filename, sep=sep_, decimal=dec_).dropna()
        self.data_array = self.data.values[:,1:].astype(dtype=float)

    def augment_dataset(self, target):
        '''
        Augment the given dataset with the supplied
        target label value.
        :param:
            target: Target label 
        :returns:
            self.data_array: Augmented dataset
        '''
        self.data_array = np.hstack((self.data_array, np.full((self.data_array.shape[0], 1), target)))
        return self.data_array

    def visualise_data(self):
        '''
            Read a given file into a pandas dataframe.
            Then, plot the various features against 
            values of frequency.
            :param: 
                None
            :returns:
                None
        '''
        self.data.plot(x='nm', y=self.data.columns[1:], kind='line', figsize=(20, 15))

    def extract_features(self, no_of_features):
        '''
        Extract usable features from the dataset.
        :param:
            no_of_features: Number of features to be extracted
        :returns:
            extracted_features: Extracted features
        '''
        X = self.data_array[:, :-1]
        y = self.data_array[:, -1]

        test = SelectKBest(score_func=f_classif, k=no_of_features)
        fit = test.fit(X, y)
        extracted_features = np.hstack((fit.transform(X), y.reshape(self.data_array.shape[0], 1)))
    
        return extracted_features 
    
    def align_dataset(self, other, update_other=False, debug=False):
        '''
            Align self to other dataframe, by appropriately fixing
            missing terms.
            :param:
                other: DataManipulation object to be aligned to
                update_other: Flag to update other, default False
                debug: Enable debug print statements, default False
            :returns:
                None
        '''
        self_ = self.data['nm'].values
        self_min, self_max = self_[0], self_[-1]
        
        other_ = other.data['nm'].values
        other_min, other_max = other_[0], other_[-1]
        
        min_index = self_min if self_min > other_min else other_min
        max_index = self_max if self_max < other_max else other_max
        
        self.data_array = self.data[self.data['nm'].isin(range(min_index, max_index+1, 10))]\
                                    .values[:,1:].astype(dtype=float)
        if update_other:
            other.data_array = other.data[other.data['nm'].isin(range(min_index, max_index+1, 10))]\
                                          .values[:,1:].astype(dtype=float)
        
        if debug:
            print ("Self: ", self_min, self_max, self.data_array.shape)
            print ("Other: ", other_min, other_max, other.data_array.shape)
            print ("Min/max: ", min_index, max_index)
        
    def cumulative_dataset(data_):
        '''
            Cumulate the data into one single data
            form.
            :param: 
                data_: Datasets to be accumulated
            :returns:
                dataset_: Complete dataset
        '''
        dataset_ = np.empty((0, data_[0].shape[1]))

        for i in data_:
            dataset_ = np.vstack((dataset_, i))

        return dataset_

if __name__ == '__main__':

    test = DataManipulation(ARCHIV+'Stoff.csv')
    test.augment_dataset(target=0)
    test_features = test.extract_features(no_of_features=6)
    print (test_features.shape)
