import time
import itertools
import numpy as np

class Apriori:
    def __init__(self, baskets, support_threshold):
        self.baskets = baskets
        self.support_threshold = support_threshold
        self.freq_itemsets = []
        self.itemsets2freq = {}

    def get_freq_singletons(self):
        for basket in self.baskets:
            for item in basket:
                if item not in self.itemsets2freq:
                    self.itemsets2freq[item] = 1
                else:
                    self.itemsets2freq[item] += 1
        for item, freq in self.itemsets2freq.items():
            if freq >= self.support_threshold:
                self.freq_itemsets.append(item)
        return self.freq_itemsets

    def subset_in_basket(self, candidate_sets, basket, k):
        return [sets for sets in itertools.combinations(basket, k) if sets in candidate_sets]

    def get_k_freq_itemsets(self, L, k):
        unique_items = np.unique(np.hstack(np.array(L)))
        Ck = list(itertools.combinations(unique_items, k))
        freq = {candidate : 0 for candidate in Ck}
        for basket in self.baskets:
            for c in Ck:
                if set(c).issubset(set(basket)):
                    freq[c] += 1
        Lk = [candidate for candidate, count in freq.items() if count >= self.support_threshold]
        return Lk

    def prune_dataset(self, itemsets):
        new_baskets = []
        for basket in self.baskets:
            for itemset in itemsets:
                if type(itemset) == int:
                    if itemset in basket:
                        new_baskets.append(basket)
                        break
                else:
                    if set(itemset).issubset(set(basket)):
                        new_baskets.append(basket)
                        break
        self.baskets = new_baskets

    def apriori(self):
        L1 = self.get_freq_singletons()
        print("Number of frequent singletons:", len(L1))
        print('Singletons:', L1)
        self.prune_dataset(L1)
        print('length of dataset after singleton pruning:', len(self.baskets))
        k = 2
        L_prev = L1
        while True:
            Lk = self.get_k_freq_itemsets(L_prev, k)
            if len(Lk) == 0:
                break
            print("Number of frequent itemsets of size {}: {}".format(k, len(Lk)))
            self.freq_itemsets.extend(Lk)
            self.prune_dataset(Lk)
            L_prev = Lk
            k += 1

def read_data(name):  # read in the dataset as a list of sets of transactions
    baskets = []
    with open(name, "r") as f:
        for line in f:
            lines = list(map(int, line.strip().split()))
            lines.sort()
            baskets.append(lines)
        baskets.sort()
        s = len(baskets) * 0.01
        return baskets, s

def main():
    start = time.time()
    baskets, s = read_data("T10I4D100K.dat")
    print('Number of transactions', len(baskets))
    print('Support threshold', s)
    alg = Apriori(baskets, s)
    alg.apriori()
    end = time.time()
    print("Time:", end - start)

if __name__ == "__main__":
    main()

