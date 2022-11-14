import os
import string

class Dataset: # read in the dataset and preprocess the documents
    def __init__(self, path):
        self.dataset_path = path
        self.dataset = []
    
    def preprocess(self, doc):
        # replace non-ascii characters and new lines with empty string
        doc = ''.join(['' if ord(i)>127 or i=='\n' else i for i in doc])
        # remove punctuations
        doc = doc.translate(str.maketrans('', '', string.punctuation))
        # convert to lowercase
        doc = doc.lower()
        # remove white spaces
        doc = doc.strip()
        return doc

    def read_data(self):
        for file in os.listdir(self.dataset_path): 
            with open(os.path.join(self.dataset_path, file), encoding="ISO-8859-1") as f:
                doc = f.read()
                doc = self.preprocess(doc)
                self.dataset.append(doc)

        return self.dataset

    def get_file_names(self):
        return os.listdir(self.dataset_path)

