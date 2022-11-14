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

    def get_shingle_num(self, dataset_shingles):
        for (i, doc) in enumerate(dataset_shingles):
            print('Document {}: {} shingles'.format(i, len(doc)))

class CompareSets: # compute the jaccard similarity between two sets(hashed shingles)
    def jaccard_similarity(self, set1, set2):
        re = len(set1.intersection(set2)) / len(set1.union(set2))
        return ('%.3f' % re)

class MinHashing: 
    def __init__(self, k=100):
        self.k = k
        self.r = 10000000
        self.a = np.random.randint(1, self.r, size=k)
        self.b = np.random.randint(1, self.r, size=k)

    def universal_hash(self, x, a, b):
        return ((a * x + b) % self.r) 

    def compute_doc_sig(self, doc_shingles): # perform minhashing on a document
        sig = np.full(self.k, math.inf)
        for i in range(self.k):
            for shingle in doc_shingles:
                sig[i] = min(sig[i], self.universal_hash(shingle, self.a[i], self.b[i]))
        return sig # the min shingle value for each hash function
    
    def compute_dataset_sig(self, dataset_shingles):
        sig_list = []
        for doc_shingles in dataset_shingles:
            sig_list.append(self.compute_doc_sig(doc_shingles))
        return sig_list

class CompareSignatures:
    def sig_similarity(self, sig1, sig2):
        cnt = 0
        for i in range(len(sig1)):
            if sig1[i] == sig2[i]:
                cnt+=1
        return cnt / len(sig1)

### Optional: Implement LSH
class LSH:
    def __init__(self, sig_length, threshold, busket_num):
        self.bands = 0
        self.rows = 0
        self.a = []
        self.b = []
        self.p = 10000000
        self.sig_length = sig_length
        self.threshold = threshold
        self.busket_num = busket_num
        self.candidate_pairs = set()

    def compute_settings(self): # compute the number of bands and rows given the threshold and sig_length
        min_diff = 1
        for i in range(1, self.sig_length):
            if self.sig_length % i == 0:
                bands = i
                rows = int(self.sig_length / i)
                t = np.power(1/bands, 1/rows)
                diff = abs(t - self.threshold)
                if diff < min_diff:
                    min_diff = diff
                    self.bands, self.rows = bands, rows

        self.a = np.random.randint(1, self.p, size=(self.bands, self.rows))
        self.b = np.random.randint(1, self.p, size=self.bands)
        print('Number of bands: {}, number of rows: {}'.format(self.bands, self.rows))

    def lsh_hash(self, x, b_id): # x is a vector, b_id band index
        return ((np.dot(self.a[b_id], x) + self.b[b_id]) % self.p) % self.busket_num

    def compute_candidates(self, sig_list):
        for band_idx in range(self.bands):
            busket = {}
            for (i, sig) in enumerate(sig_list):
                v = self.lsh_hash(sig[band_idx*self.rows:(band_idx+1)*self.rows], band_idx)
                if v not in busket:
                    busket[v] = [i]
                else:
                    busket[v].append(i)
            for value in busket.values():
                if len(value) > 1:
                    self.candidate_pairs.update(itertools.combinations(value, 2))
        return self.candidate_pairs

if __name__ == "__main__":
    # Read in the dataset
    dataset_preprocessor = Dataset(r"A1\dataset")
    dataset = dataset_preprocessor.read_data()
    # print(dataset[:3])

    start = time.time()

    # Shingling
    shingler = Shingling()
    dataset_shingles = shingler.create_dataset_shingles(dataset)
    shingler.get_shingle_num(dataset_shingles)
    # print('shingles',dataset_shingles[3])

    shingle_comparer = CompareSets()
    jaccard_similarity_matrix = np.zeros((len(dataset_shingles), len(dataset_shingles)))
    for i in range(len(dataset_shingles)):
        for j in range(i, len(dataset_shingles)):
           jaccard_similarity_matrix[i][j] = shingle_comparer.jaccard_similarity(set(dataset_shingles[i]), set(dataset_shingles[j]))
    print('jaccard_similarity_matrix: \n', jaccard_similarity_matrix)

    # MinHashing
    Minhasher = MinHashing()
    dataset_sig = Minhasher.compute_dataset_sig(dataset_shingles)
    # print(dataset_sig[:3])

    # CompareSignatures
    sig_comparator = CompareSignatures()
    sig_similarity_matrix = np.zeros((len(dataset_sig), len(dataset_sig)))
    for i in range(len(dataset_sig)):
        for j in range(i, len(dataset_sig)):
            sig_similarity_matrix[i][j] = sig_comparator.sig_similarity(dataset_sig[i], dataset_sig[j])
    print('signature_similarity_matrix: \n', sig_similarity_matrix)

    # LSH
    threshold = 0.25
    lshashing = LSH(100, threshold, 200)
    lshashing.compute_settings()
    candidate_pairs = lshashing.compute_candidates(dataset_sig)

    file_names = dataset_preprocessor.get_file_names()

    for pair in candidate_pairs:
        if sig_similarity_matrix[pair[0]][pair[1]] >= threshold:
            print('Candidate pair: ', file_names[pair[0]], file_names[pair[1]])
            print('Signature similarity: ', sig_similarity_matrix[pair[0]][pair[1]], '| Jaccard similarity: ', jaccard_similarity_matrix[pair[0]][pair[1]])

    end = time.time()
    print('Total runtime is ', (end - start))
    
