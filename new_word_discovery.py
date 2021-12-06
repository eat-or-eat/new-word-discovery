import math
import pandas as pd
from collections import defaultdict


class NewWordDiscovery:
    def __init__(self, data_path):
        self.data_name = data_path
        self.max_length = 5
        self.words_count = defaultdict(int)  # 统计词数
        self.left_neighbor = defaultdict(dict)  # 统计某个词的左邻词数
        self.right_neighbor = defaultdict(dict)  # 统计某个词的右邻词数
        self.ami = {}  # 平均互信息
        self.length_word_count = defaultdict(int)  # 统计某长度的总词数
        self.left_entropy = {}  # 左邻熵
        self.right_entropy = {}  # 右邻熵
        self.word_score = {}  # 词的得分
        self.scores_sort = ()  # 词降序得分

        self.load_data('./data/' + self.data_name)
        self.calc_ami()
        self.calc_entropy()
        self.calc_word_score()
        self.export_csv()

    # 统计词数和左右邻词数
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

    # 加载数据
    def load_data(self, data_path):
        with open(data_path, encoding='utf8') as f:
            for line in f:
                for length in range(1, self.max_length):
                    self.ngram(line, length)

    # 计算某词长总数
    def calc_length_word_count(self):
        for word, count in self.words_count.items():
            self.length_word_count[len(word)] += count

    # 计算互信息
    def calc_ami(self):
        self.calc_length_word_count()
        for word, count in self.words_count.items():
            p_word = count / self.length_word_count[len(word)]
            p_chars = 1
            for char in word:
                p_chars *= self.words_count[char] / self.length_word_count[1]
            self.ami[word] = math.log(p_word / p_chars, 2) / len(word)

    # 计算左右熵
    def calc_entropy(self):
        for word, neighbor_dic in self.left_neighbor.items():
            total = sum(neighbor_dic.values())
            entropy = sum(
                [-(char_count / total) * math.log(char_count / total, 2) for char_count in neighbor_dic.values()])
            self.left_entropy[word] = entropy
        for word, neighbor_dic in self.right_neighbor.items():
            total = sum(neighbor_dic.values())
            entropy = sum(
                [-(char_count / total) * math.log(char_count / total, 2) for char_count in neighbor_dic.values()])
            self.right_entropy[word] = entropy

    # 整理新词排序
    def calc_word_score(self):
        for word in self.ami:
            if len(word) == 1:  # 如果是单个字不做分词发现计算
                continue
            ami = self.ami.get(word, 1e-3)
            le = self.left_entropy.get(word, 1e-3)
            re = self.right_entropy.get(word, 1e-3)
            self.word_score[word] = ami * max(le, re)
        self.scores_sort = sorted([(word, count) for word, count in self.word_score.items()],
                                  key=lambda x: x[1],
                                  reverse=True)

    # 文件输出
    def export_csv(self):
        content = []
        for length in range(2, 5):
            for word, score in [(word, score) for word, score in self.scores_sort if len(word) == length][:10]:
                content.append([word,
                                score,
                                self.ami.get(word, 1e-3),
                                self.left_entropy.get(word, 1e-3),
                                self.right_entropy.get(word, 1e-3)])
        column = ['新词', '权重', '互信息', '左熵', '右熵']
        test = pd.DataFrame(columns=column, data=content)
        test.to_csv('./output/' + self.data_name.split('.')[0] + '.csv', encoding='utf-8_sig')  # 有BOM的utf8编码，不然office识别不了中文csv


if __name__ == '__main__':
    data_name = 'zhihu_info.txt'
    obj = NewWordDiscovery(data_name)
