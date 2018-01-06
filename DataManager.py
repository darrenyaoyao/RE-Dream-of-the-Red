from EntityPair import EntityPair

class DataManager():
    def __init__(self):
        self.sentences, self.parses = self.read_sentences()
        self.POS_id = self.POS_dic(self.parses)
        self.training_entitypairs, self.relations = self.get_training_entitypairs()
        self.testing_entitypairs = self.get_testing_entitypairs()

    def read_sentences(self):
        end_symbols = ['。', '！', '？']
        quotation_marks = ['「', '」']
        sentences = []
        parses = []

        with open('data/Dream_of_the_Red_Chamber_seg.txt', 'rb') as f:
            lines = f.readlines()
            for line in lines:
                s = []
                parse = []
                line = line.decode('utf-8').split()
                for c in line:
                    c = c.split('_')
                    if c[0] in quotation_marks:
                        continue
                    s.append(c[0])
                    parse.append(c[1])
                    if c[0] in end_symbols:
                        sentences.append(s)
                        parses.append(parse)
                        s = []
                        parse = []
        return sentences, parses

    def get_training_entitypairs(self):
        entitypairs = []
        relations = []
        with open('data/train.txt', 'rb') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.decode('utf-8').split()
                entitypairs.append(EntityPair(line[0], line[1], line[2], line[3]))
                if line[3] not in relations:
                    relations.append(line[3])
        return entitypairs, relations

    def get_testing_entitypairs(self):
        entitypairs = []
        with open('data/test.txt', 'rb') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.decode('utf-8').split()
                entitypairs.append(EntityPair(line[0], line[1], line[2], line[3]))
        return entitypairs

    def POS_dic(self, parses):
        idx = 0
        pos_dic = {}
        for parse in parses:
            for POS in parse:
                if POS not in pos_dic:
                    pos_dic[POS] = idx
                    idx += 1
        return pos_dic
