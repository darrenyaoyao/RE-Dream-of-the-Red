class EntityPair():
    def __init__(self, id_, entity1, entity2, relation):
        self.id = id_
        self.e1 = entity1
        self.e2 = entity2
        self.relation = relation
        self.sentences = []

    def add_sentence(self, sentence):
        self.sentences.append(sentence)
