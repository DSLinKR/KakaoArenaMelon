# Import data
import itertools
from itertools import chain
from tqdm.auto import tqdm
import os
import re
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import FastText
from khaiii import KhaiiiApi
api = KhaiiiApi()

# module
from blend_final import blending_final


def cold_start(train_set, test_set):
    train = train_set
    test = test_set

    # train, test tokenization
    test['plylst_title'] = test['plylst_title'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    newtest = test[test['plylst_title']!=''] # train, test에 플레이리스트 제목이 공백인데 length가 0이 아닌 것들이 있음
    newtest = newtest[newtest['plylst_title']!=' ']
    newtest = newtest[newtest['plylst_title']!='  ']
    newtest = newtest[newtest['plylst_title']!='   ']
    newtest = newtest[newtest['plylst_title']!='    ']
    newtest = newtest[newtest['plylst_title']!='     ']
    newtest = newtest[newtest['plylst_title']!='      ']
    newtest = newtest[newtest['plylst_title']!='       ']
    newtest = newtest[newtest['plylst_title']!='        ']
    newtest = newtest[newtest['plylst_title']!='         ']
    newtest = newtest[newtest['plylst_title']!='          ']
    newtest = newtest[newtest['plylst_title']!='           ']
    newtest = newtest[newtest['plylst_title']!='            ']
    newtest = newtest[newtest['plylst_title']!='             ']
    newtest = newtest[newtest['plylst_title']!='              ']
    newtest = newtest[newtest['plylst_title']!='               ']
    newtest = newtest[newtest['plylst_title']!='                ']
    newtest = newtest[newtest['plylst_title']!='                 ']
    newtest = newtest[newtest['plylst_title']!='                  ']
    newtest = newtest[newtest['plylst_title']!='                   ']
    newtest = newtest[newtest['plylst_title']!='                    ']
    newtest = newtest[newtest['plylst_title']!='                     ']
    newtest = newtest[newtest['plylst_title']!='                      ']

    train['plylst_title'] = train['plylst_title'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    newtrain = train[train['plylst_title']!='']
    newtrain = newtrain[newtrain['plylst_title']!=' ']
    newtrain = newtrain[newtrain['plylst_title']!='  ']
    newtrain = newtrain[newtrain['plylst_title']!='   ']
    newtrain = newtrain[newtrain['plylst_title']!='    ']
    newtrain = newtrain[newtrain['plylst_title']!='     ']
    newtrain = newtrain[newtrain['plylst_title']!='      ']
    newtrain = newtrain[newtrain['plylst_title']!='       ']
    newtrain = newtrain[newtrain['plylst_title']!='        ']
    newtrain = newtrain[newtrain['plylst_title']!='         ']
    newtrain = newtrain[newtrain['plylst_title']!='          ']
    newtrain = newtrain[newtrain['plylst_title']!='           ']
    newtrain = newtrain[newtrain['plylst_title']!='            ']
    newtrain = newtrain[newtrain['plylst_title']!='             ']
    newtrain = newtrain[newtrain['plylst_title']!='              ']
    newtrain = newtrain[newtrain['plylst_title']!='               ']
    newtrain = newtrain[newtrain['plylst_title']!='                ']
    newtrain = newtrain[newtrain['plylst_title']!='                 ']
    newtrain = newtrain[newtrain['plylst_title']!='                  ']
    newtrain = newtrain[newtrain['plylst_title']!='                   ']
    newtrain = newtrain[newtrain['plylst_title']!='                    ']
    newtrain = newtrain[newtrain['plylst_title']!='                     ']
    newtrain = newtrain[newtrain['plylst_title']!='                      ']
    newtrain = newtrain[newtrain['plylst_title']!='                       ']
    newtrain = newtrain[newtrain['plylst_title']!='                        ']
    newtrain = newtrain[newtrain['plylst_title']!='                         ']
    newtrain = newtrain[newtrain['plylst_title']!='                          ']
    newtrain = newtrain[newtrain['plylst_title']!='                           ']
    newtrain = newtrain[newtrain['plylst_title']!='                            ']

    stopwords = ['게','하','야','의','가','이','은','들','는','좀','잘','걍','과','도','을','를','으로','자','에','와','한','하다',
                '고','에','때','ㄴ','ㄹ','여','어','에서','까지','어요','아요','다','ㅋ','ㅎ','ㅠ','ㅜ','근데','더', '네', '요',
                 '다가', '해서', '아','곡','오', '노래', '음악', '뮤직']

    def morphs(s):
        token_t = []
        for word in api.analyze(s):
            for morph in word.morphs:
                if morph.lex not in stopwords:
                    token_t.append((morph.lex))
        return token_t

    tokenized = []
    for sentence in newtrain['plylst_title']:  # train tokenization
        temp = morphs(sentence)
        temp = [word for word in temp if not word in stopwords]
        tokenized.append(temp)

    tokenized_test = []
    for sentence in newtest['plylst_title']:  # test tokenization
        temp1 = morphs(sentence)
        temp1 = [word for word in temp1 if not word in stopwords]
        tokenized_test.append(temp1)

    tokenized_all = tokenized + tokenized_test  # join tokenized lists
    doc_token_list = [' '.join(tokens) for tokens in tokenized]  # tokenized의 각 token list: token끼리 join하기
    test_token_list = [' '.join(tokens) for tokens in tokenized_test]  # tokenized_test의 token끼리 join하기
    full_token_list = [' '.join(tokens) for tokens in tokenized_all]  # tokenized_all의 token끼리 join하기


    # Model: FastText weighted Tf-Idf
    model = FastText(tokenized_all, size=32, window=3, min_count=1)
    words = list(model.wv.vocab)
    tf_idf_vect = TfidfVectorizer()

    final_tf_idf = tf_idf_vect.fit_transform(full_token_list)
    tfidf_feat = tf_idf_vect.get_feature_names()

    tqdm.pandas()
    tfidf_sent_vectors = [];  # the tfidf-ft for each title is stored in this list
    row=0;
    errors=0
    for sent in tqdm(tokenized_all):  # for each title
        sent_vec = np.zeros(32)  # as word vectors are of zero length
        weight_sum =0;  # num of words with a valid vector in the sentence/review
        for word in sent:  # for each word in a review/sentence
            if word in words and word in tfidf_feat:
                vec = model.wv[word]
                tfidf = final_tf_idf[row, tfidf_feat.index(word)]
                sent_vec += (vec * tfidf)
                weight_sum += tfidf
        if weight_sum != 0:
            sent_vec /= weight_sum
        tfidf_sent_vectors.append(sent_vec)
        row += 1


    # Make new data with train + test
    frames = [newtrain, newtest]
    data = pd.concat(frames)
    data.reset_index(drop=False, inplace=True)
    data.columns = ['idx', 'tags', 'id', 'plylst_title', 'songs', 'like_cnt', 'updt_date']


    # Function Construction
    ## 1. find_similar_titles
    from sklearn.metrics.pairwise import cosine_similarity
    df = pd.DataFrame(tfidf_sent_vectors)

    def find_similar_titles(plylst_id, n):
        df['distances'] = cosine_similarity(df, df.iloc[plylst_id:plylst_id+1])
        n_largest = df['distances'].nlargest(n+1) # this contains the original plylst itself, so we need n+1
        temp = list(n_largest.index)
        temp.append(plylst_id)
        temp2 = set(temp) - set([plylst_id])
        return list(temp2)

    ## 2. find_songs
    def find_songs(plylst_id):  # function that finds all songs from find_similar_titles (빈도 순)
        ids = find_similar_titles(plylst_id, 100)
        candidates = []
        for i in ids:
            a = data.loc[i].songs
            candidates.append(a)
        merged = [item for sublist in candidates for item in sublist]
        c = Counter(merged)
        l = c.most_common(100)
        song = [sublist[0] for sublist in l]
        return song

    ## 3. find_tags
    def find_tags(plylst_id):  # function that finds all tags from find_similar_titles (빈도 순)
        ids = find_similar_titles(plylst_id, 100)
        candidates = []
        for i in ids:
            a = data.loc[i].tags
            candidates.append(a)
        merged = [item for sublist in candidates for item in sublist]
        c = Counter(merged)
        l = c.most_common(10)
        tag = [sublist[0] for sublist in l]
        if len(tag)!=10:
            ids = find_similar_titles(plylst_id, 150)
            candidates = []
            for i in ids:
                a = data.loc[i].tags
                candidates.append(a)
            merged = [item for sublist in candidates for item in sublist]
            c = Counter(merged)
            l = c.most_common(10)
            tag = [sublist[0] for sublist in l]
        return tag


    # Input cold start data
    results = pd.DataFrame(blending_final(train, test))

    blank_test = newtest[newtest.tags.apply(len)==0]
    blank_test = blank_test[blank_test.songs.apply(len)==0]
    blank_test = blank_test[blank_test.plylst_title.apply(len)!=0]
    cs = list(blank_test.index)

    ## song, tag을 채우기 위한 indices_song 찾기
    data1 = data.iloc[109421:]
    idxs = []
    for i in cs:
        row = data1.loc[data1['idx'] == i]
        a = row.index.tolist()
        idxs.append(a)

    indices_song = [item for sublist in idxs for item in sublist]

    answer1 = []
    for i in tqdm(indices_song):
        songs = find_songs(i)
        answer1.append(songs)

    answer2 = []
    for i in tqdm(indices_song):
        tags = find_tags(i)
        answer2.append(tags)


    # results + cold start 답으로도 안 채워진 아예 빈 플레이리스트들을 popular songs, tags로 채워넣기
    l = set(test.id) - set(results.id)
    b = l - set(blank_test.id)

    song_count_dict = Counter([x for y in train.songs for x in y])
    tag_count_dict = Counter([x for y in train.tags for x in y])

    song_count_dict_sorted_by_values ={k : v for k, v in sorted(song_count_dict.items(), key=lambda x: x[1], reverse=True)}
    tag_count_dict_sorted_by_values ={k : v for k, v in sorted(tag_count_dict.items(), key=lambda x: x[1], reverse=True)}

    popular_songs = list(song_count_dict_sorted_by_values.keys())[:100]
    popular_tags = list(tag_count_dict_sorted_by_values.keys())[:10]


    returnval = []
    for _id, rec, tag_rec in zip(results.id, results.songs, results.tags):
        returnval.append({
            "id": _id,
            "songs": list(map(int, rec[:100])),
            "tags": tag_rec[:10]
        })

    for _id, rec, tag_rec in zip(blank_test.id, answer1, answer2):
        returnval.append({
            "id": _id,
            "songs": list(map(int,rec)),
            "tags": tag_rec
        })

    for _id in b:
        returnval.append({
            "id": _id,
            "songs": list(map(int,popular_songs)),
            "tags": popular_tags
        })

    return returnval
