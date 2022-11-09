import os
import json

class Dataset:
    def __init__(self, path):
        self.dataset_path = path
        self.dataset = []

    def read_data(self):
        for file in os.listdir(self.dataset_path): # read in json files from the given path
            with open(os.path.join(self.dataset_path, file), 'r') as f:
                self.dataset.append(json.load(f, encoding='utf-8'))  # append the json file to the dataset
    
    def preprocess(self, dataset): # preprocess each file in the datdaset
        for i in range(len(dataset)):
            dataset[i] = dataset[i]['text'].lower()
            # remove non-ascii characters
            dataset[i] = ''.join([c for c in dataset[i] if ord(c) < 128])
            # remove newlines
            dataset[i] = dataset[i].replace('\n', ' ')
            # remove extra spaces
            dataset[i] = ' '.join(dataset[i].split())
        return dataset

    def print_dataset(self, num):
            print(self.dataset[num])
            

