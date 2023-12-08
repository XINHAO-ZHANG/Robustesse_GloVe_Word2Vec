import numpy as np
def read_embedding(path):
    vocab = []
    word_embedding = []
    with open(path,'r',encoding='utf-8') as f:
        for i in f.readlines():
            cut_data = i.split()
            vocab.append(cut_data[0])
            word_embedding.append(list(map(float,cut_data[1:])))

    word_to_idx = {}
    for number, word in enumerate(vocab):
        word_to_idx[word] = number
    return vocab, word_embedding, word_to_idx

def simcos(a,b):
    dot = sum(a*b)
    mod_a = sum(a**2)**0.5
    mod_b = sum(b**2)**0.5
    return dot/(mod_a*mod_b)

def calculate_sim(word_embedding, word_to_idx, word_name):
    np_embedding = np.array(word_embedding)

    all_sim = []
    for i in range(len(word_embedding)):
        all_sim.append([simcos(np_embedding[word_to_idx[word_name]],np_embedding[i]),i])
    all_sim.sort(reverse=True,key=lambda x:x[0])
    return all_sim


def get_top20_words(sims, vocab):
    top20 = sims[1:21]
    words = [vocab[i[1]] for i in top20]
    return words


glove_vovab, glove_embedding, glove_to_idx = read_embedding('./vectors_glove.txt')
word2vec_vovab, word2vec_embedding, word2vec_to_idx = read_embedding('./vectors_word2vec.txt')
with open('vocab_train.txt', 'r', encoding='utf-8') as f:
    top = []
    for i in range(20):
        line = f.readline()
        top.append(line.split(' ')[0])

for i in top:
    glove_sim = calculate_sim(glove_embedding, glove_to_idx, i)
    word2vec_sim = calculate_sim(word2vec_embedding, word2vec_to_idx, i)
    glove = get_top20_words(glove_sim, glove_vovab)
    w2v = get_top20_words(word2vec_sim, word2vec_vovab)
    print(f'word_name:{i} \nglove_top20_words:{glove} \nw2v_top20_words:{w2v}')
