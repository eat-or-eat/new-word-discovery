import math
from collections import defaultdict


class NewWordDiscovery:
    def __init__(self, data_path):
        self.max_length = 5
        self.words_count = defaultdict(int)
        self.left_neighbor = defaultdict(dict)
        self.right_neighbor = defaultdict(dict)
        self.ami = {}
        self.length_word_count = defaultdict(int)
        self.left_entropy = {}
        self.right_entropy = {}
        self.word_score = {}

        self.load_data(data_path)
        self.calc_ami()
        self.calc_entropy()
        self.calc_word_score()

    def ngram(self, line, length):
        for i in range(len(line) - length + 1):
            word = line[i:i + length]
            self.words_count[word] += 1
            if i - 1 > 0:
                char = line[i - 1]
                self.left_neighbor[word][char] = self.left_neighbor[word].get(char, 0) + 1
            if i + length < len(line):
                char = line[i + length]
                self.right_neighbor[word][char] = self.right_neighbor[word].get(char, 0) + 1

    def load_data(self, data_path):
        with open(data_path, encoding='utf8') as f:
            for line in f:
                for length in range(1, self.max_length):
                    self.ngram(line, length)

    def calc_length_word_count(self):
        for word, count in self.words_count.items():
            self.length_word_count[len(word)] += count

    def calc_ami(self):
        self.calc_length_word_count()
        for word, count in self.words_count.items():
            p_word = count / self.length_word_count[len(word)]
            p_chars = 1
            for char in word:
                p_chars *= self.words_count[char] / self.length_word_count[1]
            self.ami[word] = math.log(p_word / p_chars, 2) / len(word)

    def calc_entropy(self):
        for word, neighbor_dic in self.left_neighbor.items():
            total = sum(neighbor_dic.values())
            entropy = sum([-(char_count / total) * math.log(char_count / total, 2) for char_count in neighbor_dic.values()])
            self.left_entropy[word] = entropy
        for word, neighbor_dic in self.right_neighbor.items():
            total = sum(neighbor_dic.values())
            entropy = sum([-(char_count / total) * math.log(char_count / total, 2) for char_count in neighbor_dic.values()])
            self.right_entropy[word] = entropy

    def calc_word_score(self):
        for word in self.ami:
            ami = self.ami.get(word, 1e-3)
            le = self.left_entropy.get(word, 1e-3)
            re = self.right_entropy.get(word, 1e-3)
            self.word_score[word] = ami * max(le, re)


if __name__ == '__main__':
    obj = NewWordDiscovery('./data/tianchi_match.csv')
    scores_sort = sorted([(word, count) for word, count in obj.word_score.items()], key=lambda x: x[1], reverse=True)
    print([x for x, c in scores_sort if len(x) == 1][:10])
    print([x for x, c in scores_sort if len(x) == 2][:10])
    print([x for x, c in scores_sort if len(x) == 3][:10])
    print([x for x, c in scores_sort if len(x) == 4][:10])
