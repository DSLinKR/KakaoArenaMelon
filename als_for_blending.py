import pandas as pd
import numpy as np
from implicit.evaluation import  *
from implicit.als import AlternatingLeastSquares as ALS
from implicit.bpr import BayesianPersonalizedRanking as BPR
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tqdm.auto import tqdm
from collections import Counter
from itertools import groupby
from scipy.sparse import *
import scipy.sparse


def als_for_blending(train_set, test_set) :
    tr = train_set
    te = test_set

    ret = []
    for tag in tr.tags.tolist():
        ret += tag
    r = dict(Counter(ret))
    r = sorted(r.items(), key=lambda x: -x[1])

    tr_songs = tr.songs.tolist()
    te_songs = te.songs.tolist()
    tr_tags = tr.tags.tolist()
    te_tags = te.tags.tolist()
    te_ids = te.id.tolist()

    
    tr = []
    iid_to_idx = {}
    tag_to_idx = {} 
    idx = 0

    # train의 songs에게 새로운 인덱스 부여
    for i, l in enumerate(tr_songs):
        view = l # item
        for item_id in view:
            if item_id not in iid_to_idx:
                iid_to_idx[item_id] = idx
                idx += 1
        view = [iid_to_idx[x] for x in view]
        tr.append(view)
        
    # train의 tags에게 새로운 인덱스 부여
    # 노래 마지막 인덱스 이후의 인덱스 부여
    idx = 0
    n_items = len(iid_to_idx)
    n_tags = len(tag_to_idx)
    for i, tags in enumerate(tr_tags):
        for tag in tags:
            if tag not in tag_to_idx:
                tag_to_idx[tag] = n_items + idx
                idx += 1
        tr[i].extend([tag_to_idx[x] for x in tags])
        

    # test 노래와 태그에도 똑같이!
    # test에만 있는 노래나 태그 등장할 경우 continue 써서 무시함
    n_tags = len(tag_to_idx)
    te = []

    idx = 0
    for i, l in enumerate(te_songs):
        view = l
        ret = [] 
        for item_id in view:
            if item_id not in iid_to_idx:
                continue
            ret.append(iid_to_idx[item_id])
        te.append(ret)
    idx = 0
    for i, tags in enumerate(te_tags):
        ret = []
        for tag in tags:
            if tag not in tag_to_idx:
                continue
            ret.append(tag)
        te[i].extend([tag_to_idx[x] for x in ret])
        

    # 나중에 추천된 노래와 태그를 다시 원래 index로 바꾸기 위해 만드는 dictionary들
    idx_to_iid = {x:y for(y,x) in iid_to_idx.items()}
    idx_to_tag = {(x - n_items):y for(y,x) in tag_to_idx.items()}

    # [y for x in list for y in x] -> 원소로 list를 담고있는 큰 list에서 안에 있는 리스트의 원소를 빼내는 작업
    # cold는 tr의 모든 노래 + 모든 태그
    cold = np.array([y for x in tr for y in x])

    # rowd는 tr의 모든 플레이리스트 index를 그 속에 속한 내용만큼 반복
    # rowd[:20] = [0,0,...,0]
    reprow=[]
    for i in range(len(tr)):
        reprow.append(np.repeat(i, len(tr[i])))
    rowd = np.array([y for x in reprow for y in x])

    # infod 는 cold만큼, 즉 모든 노래 + 모든 태그 만큼 1을 만든 list
    infod = list(map(int, np.repeat(1,len(cold))))

    # tr을 csr 로 만든다
    tr_csr = csr_matrix((infod,(rowd, cold)), shape=(len(tr), n_tags + n_items))

    # test에도 똑같이 한다
    cold2 = np.array([y for x in te for y in x])

    reprow2=[]
    for i in range(len(te)):
        reprow2.append(np.repeat(i, len(te[i])))
    rowd2 = np.array([y for x in reprow2 for y in x])

    infod2 = list(map(int, np.repeat(1,len(cold2))))

    te_csr = csr_matrix((infod2,(rowd2, cold2)), shape=(len(te), n_tags + n_items))

    # 위를 te로, 아래를 tr로 갖는 csr 매트릭스 완성
    r = scipy.sparse.vstack([te_csr, tr_csr])
    r = csr_matrix(r)

    ## Cold Start 빼기
    te_data = test_set

    no_tag_data = te_data[ te_data.tags.apply(len) == 0 ]
    cold_start_data = no_tag_data[ no_tag_data.songs.apply(len) == 0 ]

    # 원래 cold_start 였던 행의 인덱스
    cs_idx = list(cold_start_data.index)

    # r_cs Cold Start 사라진 CSR
    r_cs = r[np.where(r.sum(axis=1))[0] , : ]

    # te_ids_cs : Cold Start 아닌 플레이리스트의 id 만 남김
    idx_to_teid = {i:j for (i,j) in zip(range(len(te)), te_ids)}
    te_ids_cs = [i for i in te_ids if i not in [idx_to_teid[j] for j in cs_idx] ]

    # Modeling
    als_model = ALS(factors=128, regularization=0.08, use_gpu=False) # colab에서 돌릴 경우 use_gpu=True 옵션 추가
    als_model.fit(r_cs.T * 15.0)

    item_model = ALS(factors=128, use_gpu=True)
    tag_model = ALS(factors=128, use_gpu=True)
    item_model.user_factors = als_model.user_factors
    tag_model.user_factors = als_model.user_factors

    item_model.item_factors = als_model.item_factors[:n_items]
    tag_model.item_factors = als_model.item_factors[n_items:]

    item_rec_csr = tr_csr[:, :n_items]
    tag_rec_csr = tr_csr[:, n_items:]

    item_ret = []
    tag_ret = []
    
    for u in tqdm(range(te_csr.shape[0])):
        item_rec = item_model.recommend(u, item_rec_csr, N=500)
        item_rec = [idx_to_iid[x[0]] for x in item_rec]
        tag_rec = tag_model.recommend(u, tag_rec_csr, N=50)
        tag_rec = [idx_to_tag[x[0]] for x in tag_rec if x[0] in idx_to_tag]
        item_ret.append(item_rec)
        tag_ret.append(tag_rec)

    item_rank = []
    tag_rank = []

    for i in tqdm(range(te_csr.shape[0])) :
        item_rank_vec = item_model.rank_items(i, te_csr,  [iid_to_idx[x] for x in item_ret[i]]  )
        item_mu = np.mean([j for i,j in item_rank_vec])
        item_std = np.std([j for i,j in item_rank_vec])

        item_rank_dict = {}
        for idx, rank in item_rank_vec : 
            item_rank_dict[idx_to_iid[idx]] = float( (rank - item_mu) / item_std )
        item_rank.append(item_rank_dict)

        tag_rank_vec = tag_model.rank_items(i, te_csr, [ (tag_to_idx[x] - n_items) for x in tag_ret[i][:50] ] )
        tag_mu = np.mean([j for i,j in tag_rank_vec])
        tag_std = np.std([j for i,j in tag_rank_vec])
        
        tag_rank_dict = {}
        for idx, rank in tag_rank_vec :
            tag_rank_dict[idx_to_tag[idx]] = float( (rank - tag_mu) / tag_std )
        tag_rank.append(tag_rank_dict)

    # 플레이리스트 당 노래 500개, 태그 50개 들어있는 리스트 생성
    returnval = []
    for _id, rec, tag_rec, songs_score, tags_score in zip(te_ids_cs, item_ret, tag_ret, item_rank, tag_rank): # 여기 바뀜!
        returnval.append({
            "id": _id,
            "songs": rec[:500],
            "tags": tag_rec[:50],
            "songs_score" : songs_score,
            "tags_score" : tags_score
        })

    return returnval
    # import json
    # with open('als_final.json', 'w', encoding='utf-8') as f:
    #    f.write(json.dumps(returnval, ensure_ascii=False))