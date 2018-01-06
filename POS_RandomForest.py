import gensim
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ThreadPoolExecutor
from DataManager import DataManager

datamanager = DataManager()

sentences = datamanager.sentences
POSes = datamanager.parses
POS_id = datamanager.POS_id
entitypairs = datamanager.training_entitypairs
testing_entitypairs = datamanager.testing_entitypairs
relations = datamanager.relations
document = []

def check_entity_in_words(entity, words):
    if entity in words:
        return True
    elif len(entity) == 3 and entity[1:] in words:
        return True
    else:
        return False


def search_relation_sentence(entitypair):
    context = []
    context_pos = []
    e1_first_sentence = []
    e1_first_pos = []
    e2_first_sentence = []
    e2_first_pos = []
    for i, words in enumerate(sentences):
        if len(context) > 3:
            context.pop(0)
        context.append(words)
        context_pos.append(POSes[i])
        if check_entity_in_words(entitypair.e1, words) and check_entity_in_words(entitypair.e2, words):
            entitypair.add_sentence(words)
            entitypair.add_pos(POSes[i])
        elif check_entity_in_words(entitypair.e1, [word for words in context for word in words]) \
             and check_entity_in_words(entitypair.e2, [word for words in context for word in words]):
            entitypair.add_sentence([word for words in context for word in words])
            entitypair.add_pos([pos for poses in context_pos for pos in poses])
        else:
            if check_entity_in_words(entitypair.e1, words) and len(e1_first_sentence) == 0:
                e1_first_sentence = words
                e1_first_pos = POSes[i]
            if check_entity_in_words(entitypair.e2, words) and len(e2_first_sentence) == 0:
                e2_first_sentence = words
                e2_first_pos = POSes[i]
    if len(entitypair.sentences) == 0:
        entitypair.add_sentence(e1_first_sentence+e2_first_sentence)
        entitypair.add_pos(e1_first_pos+e2_first_pos)
    return entitypair.sentences

print("Start Word2Vec")
model = gensim.models.Word2Vec(sentences, size=300, min_count=1, workers=4)
print("Finish Word2Vec")

training_x = []
training_y = []
with ThreadPoolExecutor(max_workers=20) as executor:
    for entitypair, entity_sentences in zip(entitypairs, executor.map(search_relation_sentence, entitypairs)):
        entitypair_feature = np.zeros(300)
        entitypair_pos_feature = np.zeros(61)
        count = 0
        for i, sentence in enumerate(entity_sentences):
            for j, word in enumerate(sentence):
                if word in model.wv:
                    entitypair_feature += model.wv[word]
                    count += 1
                    entitypair_pos_feature[POS_id[entitypair.poses[i][j]]] += 1
        entitypair_feature = entitypair_feature/count
        entitypair_pos_feature = entitypair_pos_feature/count
        training_x.append(np.asarray(np.hstack([entitypair_feature, entitypair_pos_feature])))
        training_y.append(np.asarray(relations.index(entitypair.relation)))


clf = RandomForestClassifier(n_estimators = 400)
clf.fit(np.asarray(training_x), np.asarray(training_y))

print("Finish Training")

total = 0
correct = 0
print("Start Testing")
with ThreadPoolExecutor(max_workers=20) as executor:
    for testing_entitypair, testing_entity_sentences in zip(testing_entitypairs, executor.map(search_relation_sentence, testing_entitypairs)):
        entitypair_feature = np.zeros(300)
        entitypair_pos_feature = np.zeros(61)
        count = 0
        for i, sentence in enumerate(testing_entity_sentences):
            for j, word in enumerate(sentence):
                if word in model.wv:
                    entitypair_feature += model.wv[word]
                    count += 1
                    entitypair_pos_feature[POS_id[testing_entitypair.poses[i][j]]] += 1
        entitypair_feature = entitypair_feature/count
        entitypair_pos_feature = entitypair_pos_feature/count
        r = clf.predict(np.hstack([entitypair_feature, entitypair_pos_feature]).reshape(1, -1))[0]
        total += 1
        if testing_entitypair.relation == relations[r]:
            correct += 1
        print(testing_entitypair.e1, testing_entitypair.e2, 'Relation:', relations[r])
print('Acc:', correct/total)
