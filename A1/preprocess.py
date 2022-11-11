import os
import string

class Dataset: # read in the dataset and preprocess the documents
    def __init__(self, path):
        self.dataset_path = path
        self.dataset = []

    def read_data(self):
        for file in os.listdir(self.dataset_path): 
            with open(os.path.join(self.dataset_path, file), encoding='ISO-8859-1') as f:
                self.dataset.append(f.read())  
    
    def preprocess(self): # preprocess each file in the datdaset
        for i in range(len(self.dataset)):
            self.dataset[i] = self.dataset[i].lower()
            # remove non-ascii characters
            self.dataset[i] = ''.join([c for c in self.dataset[i] if ord(c) < 128])
            # remove newlines
            self.dataset[i] = self.dataset[i].replace('\n', ' ')
            # remove extra spaces
            self.dataset[i] = ' '.join(self.dataset[i].split())
            # remove punctuation
            self.dataset[i] = self.dataset[i].translate(str.maketrans('', '', string.punctuation))

    def get_dataset(self, k): # return the first num files in the dataset
        for i in range(k):
            print('Document {}:'.format(i))
            print(self.dataset[i])
        return self.dataset[:k]
            

