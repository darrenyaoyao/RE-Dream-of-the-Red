import gensim
import jieba
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ThreadPoolExecutor
from DataManager import DataManager

jieba.set_dictionary('dict.txt.big')
datamanager = DataManager()

sentences = datamanager.sentences
entitypairs = datamanager.training_entitypairs
testing_entitypairs = datamanager.testing_entitypairs
relations = datamanager.relations
document = []

def check_entity_in_words(entity, words):
    if entity in words or entity[1:] in words:
        return True
    else:
        return False

def search_relation_sentence(entitypair):
    context = []
    e1_first_sentence = []
    e2_first_sentence = []
    for sentence in sentences:
        words = sentence
        if len(context) > 3:
            context.pop(0)
        context.append(words)
        if check_entity_in_words(entitypair.e1, words) and check_entity_in_words(entitypair.e2, words):
            entitypair.add_sentence(words)
            break
        elif check_entity_in_words(entitypair.e1, [word for words in context for word in words]) \
             and check_entity_in_words(entitypair.e2, [word for words in context for word in words]):
            entitypair.add_sentence([word for words in context for word in words])
            break
        else:
            if check_entity_in_words(entitypair.e1, words) and len(e1_first_sentence) == 0:
                e1_first_sentence = words
            if check_entity_in_words(entitypair.e2, words) and len(e2_first_sentence) == 0:
                e2_first_sentence = words
    if len(entitypair.sentences) == 0:
        entitypair.add_sentence(e1_first_sentence+e2_first_sentence)
    return entitypair.sentences[0]

with ThreadPoolExecutor(max_workers=20) as executor:
    for entitypair, sentence in zip(entitypairs, executor.map(search_relation_sentence, entitypairs+testing_entitypairs)):
        document.append(sentence)

print("Start Word2Vec")
model = gensim.models.Word2Vec(document, size=50, window=5, min_count=1, workers=4)
model.save('baseline_word2vec.model')
print("Finish Word2Vec")

training_x = []
training_y = []
for entitypair in entitypairs:
    try:
        e1_wv = model.wv[entitypair.e1[1:]]
    except:
        e1_wv = model.wv[entitypair.e1]
    try:
        e2_wv = model.wv[entitypair.e2[1:]]
    except:
        e2_wv = model.wv[entitypair.e2]
    training_x.append(np.asarray(e1_wv)+np.asarray(e2_wv))
    training_y.append(np.asarray(relations.index(entitypair.relation)))

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(np.asarray(training_x), np.asarray(training_y))

total = 0
correct = 0
for entitypair in testing_entitypairs:
    if (entitypair.e1 in model.wv or entitypair.e1[1:] in model.wv) \
            and (entitypair.e2 in model.wv or entitypair.e2[1:] in model.wv):
        try:
            e1_wv = model.wv[entitypair.e1[1:]]
        except:
            e1_wv = model.wv[entitypair.e1]
        try:
            e2_wv = model.wv[entitypair.e2[1:]]
        except:
            e2_wv = model.wv[entitypair.e2]
        r = clf.predict(np.asarray(e1_wv)+np.asarray(e2_wv).reshape(1, -1))[0]
    else:
        r = 3
    total += 1
    if entitypair.relation == relations[r]:
        correct += 1
    print(entitypair.e1, entitypair.e2, 'Relation:', relations[r])
print('Acc:', correct/total)


