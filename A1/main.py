from preprocess import Dataset

import itertools
import numpy as np
import math
import time
 
class Shingling: 
    def __init__(self, k=5):
        self.k = k

    def hash_shingle(self, txt):
        return hash(str(txt))

    def create_doc_shingles(self, doc):
        shingles = set()
        for i in range(len(doc) - self.k + 1):
            shingles.add(self.hash_shingle(doc[i:i+self.k]))
        return shingles

    def create_dataset_shingles(self, dataset):
        shingles_list = []
        for doc in dataset:
            shingles_list.append(self.create_doc_shingles(doc))
        return shingles_list 

class CompareSets: # compute the jaccard similarity between two sets(hashed shingles)
    def jaccard_similarity(self, set1, set2):
        re = len(set1.intersection(set2)) / len(set1.union(set2))
        return ('%.3f' % re)

class MinHashing: 
    def __init__(self, k=1000):
        self.k = k
        self.r = 1000000007
        self.a = np.random.randint(1, self.r, size=k)
        self.b = np.random.randint(1, self.r, size=k)

    def universal_hash(self, x, a, b):
        return ((a * x + b) % self.r) 

    def compute_doc_sig(self, doc_shingles):
        sig = np.full(self.k, math.inf)
        for shingle in doc_shingles:
            for i in range(self.k):
                sig[i] = min(sig[i], self.universal_hash(shingle, self.a[i], self.b[i]))
        return sig
    
    def compute_dataset_sig(self, dataset_shingles):
        sig_list = []
        for doc_shingles in dataset_shingles:
            sig_list.append(self.compute_doc_sig(doc_shingles))
        return sig_list

class CompareSignatures:
    def sig_similarity(self, sig1, sig2):
        return len([i for i in range(len(sig1)) if sig1[i] == sig2[i]]) / len(sig1)

### Optional: Implement LSH
class LSH:
    def __init__(self, b, r):
        self.b = b
        self.r = r
        self.c = 1000000007
        self.a = np.random.randint(1, self.c, size=b)
        self.b = np.random.randint(1, self.c, size=b)

    def hash_function(self, x, a, b):
        return ((a * x + b) % self.c)

    def compute_bands(self, sig):
        bands = []
        for i in range(self.b):
            band = []
            for j in range(self.r):
                band.append(self.hash_function(sig[i*self.r+j], self.a[i], self.b[i]))
            bands.append(band)
        return bands

    def compute_bands_dataset(self, sig_list):
        bands_list = []
        for sig in sig_list:
            bands_list.append(self.compute_bands(sig))
        return bands_list

    def find_similar_docs(self, bands_list, threshold):
        similar_docs = []
        for i in range(len(bands_list)):
            for j in range(i+1, len(bands_list)):
                for band1 in bands_list[i]:
                    for band2 in bands_list[j]:
                        if band1 == band2:
                            similar_docs.append((i, j))
                            break
        return similar_docs

if __name__ == "__main__":
    # Read in the dataset
    dataset_preprocessor = Dataset('A1\\dataset')
    dataset = dataset_preprocessor.read_data()
    # print(dataset[:3])

    start = time.time()

    # Shingling
    shingler = Shingling()
    dataset_shingles = shingler.create_dataset_shingles(dataset)
    # print(dataset_shingles[:3])

    shingle_comparer = CompareSets()
    jaccard_similarity_matrix = np.zeros((len(dataset_shingles), len(dataset_shingles)))
    for i in range(len(dataset_shingles)):
        for j in range(i+1, len(dataset_shingles)):
           jaccard_similarity_matrix[i][j] = shingle_comparer.jaccard_similarity(dataset_shingles[i], dataset_shingles[j])
    print('jaccard_similarity_matrix in main: \n', jaccard_similarity_matrix)

    # MinHashing
    Minhasher = MinHashing()
    dataset_sig = Minhasher.compute_dataset_sig(dataset_shingles)
    # print(dataset_sig[:3])

    # CompareSignatures
    sig_comparator = CompareSignatures()
    sig_similarity_matrix = np.zeros((len(dataset_sig), len(dataset_sig)))
    for i in range(len(dataset_sig)):
        for j in range(i+1, len(dataset_sig)):
            sig_similarity_matrix[i][j] = sig_comparator.sig_similarity(dataset_sig[i], dataset_sig[j])
    print('signature_similarity_matrix in main: \n', sig_similarity_matrix)

    end = time.time()
    print('Total runtime is ', (end - start))
    
