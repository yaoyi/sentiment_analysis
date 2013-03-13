from math import log

class TfIdf:
    def __init__(self):
        self._idf = 1.5
    def calcidf(self, lines):
        result = {}
        words = set()
        total_count = 0
        for line in lines:
            for word in line.split():
                words.add(word)
                total_count += 1
        for word in words:
            result[word] = 0
 
        for line in lines:
            for word in words:
                if word in line:
                    result[word] += 1
        for i,j in result.items():
            result[i] = log(total_count/j, 2)
        self._idf = result    
    @property
    def idf(self):
        return self._idf