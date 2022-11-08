from dataset import Dataset

import itertools
import numpy as np
import math

class Shingling: # unedited
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k

    def shingle(self, text):
        shingles = set()
        for i in range(len(text) - self.k + 1):
            shingles.add(text[i:i+self.k])
        return shingles

    def shingle_dataset(self):
        shingled_dataset = []
        for data in self.dataset:
            shingled_dataset.append(self.shingle(data))
        return shingled_dataset

    def jaccard(self, set1, set2):
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def minhash(self, set1, set2):
        return min([hash(s) for s in set1.intersection(set2)])

    def lsh(self, set1, set2):
        return hash(set1) == hash(set2)

    def compare(self, set1, set2):
        return self.jaccard(set1, set2), self.minhash(set1, set2), self.lsh(set1, set2)

    def compare_dataset(self):
        comparisons = []
        for set1, set2 in itertools.combinations(self.shingled_dataset, 2):
            comparisons.append(self.compare(set1, set2))
        return comparisons

class CompareSets: # compute the jaccard similarity between two sets(hashed shingles)
    def jaccard_similarity(self, set1, set2):
        return len(set1.intersection(set2)) / len(set1.union(set2))

class MinHashing: # unedited
    def __init__(self, k=100):
        self.k = k

class CompareSignatures:
    def sig_similarity(self, sig1, sig2):
        return len([i for i in range(len(sig1)) if sig1[i] == sig2[i]]) / len(sig1)

### Optional: Implement LSH
# class LSH:

if __name__ == "__main__":
    # Read in the dataset
    dataset = Dataset('data')
    dataset = dataset.read_data()

    # Shingling
    shingling = Shingling(dataset, 5)
    shingled_dataset = shingling.shingle_dataset()

    # Compare the shingled documents
    compare_sets = CompareSets()
    comparisons = []
    for set1, set2 in itertools.combinations(shingled_dataset, 2):
        comparisons.append(compare_sets.jaccard_similarity(set1, set2))

    print('Jaccard Similarity:', np.mean(comparisons))

    # Minhashing
    minhashing = MinHashing()
    minhashing = minhashing.minhash(shingled_dataset)

    # Compare the minhash signatures
    compare_signatures = CompareSignatures()
    comparisons = []
    for sig1, sig2 in itertools.combinations(minhashing, 2):
        comparisons.append(compare_signatures.sig_similarity(sig1, sig2))

    print('Minhash Similarity:', np.mean(comparisons))
