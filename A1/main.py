from preprocess import Dataset

import itertools
import numpy as np
import math
import time
 
class Shingling: 
    def __init__(self, k=9):
        self.k = k

    def hash_shingle(self, shingle):
        return hash(str(shingle))

    def shingle_doc(self, doc):
        shingles = set()
        for i in range(len(doc) - self.k + 1):
            shingles.add(self.hash_shingle(doc[i:i+self.k]))
        return set(shingles)

    def shingle_dataset(self, dataset):
        shingles_list = []
        for doc in dataset:
            shingles_list.append(self.shingle_doc(doc))
        return shingles_list 

class CompareSets: # compute the jaccard similarity between two sets(hashed shingles)
    def jaccard_similarity(self, set1, set2):
        return len(set1.intersection(set2)) / len(set1.union(set2))

class MinHashing: 
    def __init__(self, k=100):
        self.k = k
        self.r = 1000000007
        self.a = np.random.randint(1, self.r, size=k)
        self.b = np.random.randint(1, self.r, size=k)
        self.c = np.random.randint(1, self.r, size=k)

    def hash_function(self, x, a, b, c):
        return ((a * x + b) % c)

    def compute_sig_doc(self, doc):
        sig = np.full(self.k, math.inf)
        for shingle in doc:
            for i in range(self.k):
                sig[i] = min(sig[i], self.hash_function(shingle, self.a[i], self.b[i], self.c[i]))
        return sig
    
    def compute_sig_dataset(self, dataset):
        sig_list = []
        for doc in dataset:
            sig_list.append(self.compute_sig_doc(doc))
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
    start = time.time()
    # Read in the dataset
    dataset_preprocessor = Dataset('dataset')
    dataset = dataset_preprocessor.read_data()
    dataset = dataset_preprocessor.preprocess(dataset)
    # dataset_preprocessor.print_dataset(0)

    # Shingling
    shingler = Shingling()
    shingles_list = shingler.shingle_dataset(dataset)
    # print(shingles_list[0])

    # MinHashing
    minhasher = MinHashing()
    sig_list = minhasher.compute_sig_dataset(shingles_list)
    # print(sig_list[0])

    # CompareSignatures
    comparator = CompareSignatures()
    print(comparator.sig_similarity(sig_list[0], sig_list[1]))
    print(comparator.sig_similarity(sig_list[0], sig_list[2]))
    print(comparator.sig_similarity(sig_list[1], sig_list[2]))
    end = time.time()
    print('Total runtime is ', (end - start))
    
