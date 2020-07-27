from collections import Counter

import numpy as np
import pandas as pd

import scipy.sparse as spr
from tqdm import tqdm


def item_for_blening(train, test):
    
    # Reducing Data Size
    
    ret = []
    for tag in train.tags.tolist():
        ret += tag
    r = dict(Counter(ret))
    r = {key: value for key, value in r.items() if value < 16}
    t=list(r.keys())
    ret = []
    for song in train.songs.tolist():
        ret += song
    s = dict(Counter(ret))
    s = {key: value for key, value in s.items() if value < 8}
    s=list(s.keys())
    s = set(s)
    newDF = pd.DataFrame({'tr.songs': train.songs.map(set)})
    newDF['sparse_s'] = newDF.apply(lambda x:s, axis=1)
    newDF = newDF.assign(result=newDF['tr.songs']-newDF['sparse_s'])
    train.songs = newDF.result.map(list)
    t = set(t)
    newDF = pd.DataFrame({'tr.tags': train.tags.map(set)})
    newDF['sparse_s'] = newDF.apply(lambda x:t, axis=1)
    newDF = newDF.assign(result=newDF['tr.tags']-newDF['sparse_s'])
    train.tags = newDF.result.map(list)
    
    # Marking Index
    
    train['istrain'] = 1
    test['istrain'] = 0

    n_train = len(train)
    n_test = len(test)

    ## train + test
    plylst = pd.concat([train, test], ignore_index=True)

    ## playlist id
    plylst["nid"] = range(n_train + n_test)

    ## id <-> nid
    plylst_id_nid = dict(zip(plylst["id"],plylst["nid"]))
    plylst_nid_id = dict(zip(plylst["nid"],plylst["id"]))
    
    plylst_tag = plylst['tags']
    tag_counter = Counter([tg for tgs in plylst_tag for tg in tgs])
    tag_dict = {x: tag_counter[x] for x in tag_counter}

    tag_id_tid = dict()
    tag_tid_id = dict()
    for i, t in enumerate(tag_dict):
        tag_id_tid[t] = i
        tag_tid_id[i] = t

    n_tags = len(tag_dict)

    plylst_song = plylst['songs']
    song_counter = Counter([sg for sgs in plylst_song for sg in sgs])
    song_dict = {x: song_counter[x] for x in song_counter}

    song_id_sid = dict()
    song_sid_id = dict()
    for i, t in enumerate(song_dict):
        song_id_sid[t] = i
        song_sid_id[i] = t

    n_songs = len(song_dict)
    
    plylst['songs_id'] = plylst['songs'].map(lambda x: [song_id_sid.get(s) for s in x if song_id_sid.get(s) != None])
    plylst['tags_id'] = plylst['tags'].map(lambda x: [tag_id_tid.get(t) for t in x if tag_id_tid.get(t) != None])
    
    
    # Convert to usable format
    
    plylst_use = plylst.loc[:,['istrain','nid','updt_date','songs_id','tags_id']]
    plylst_use.loc[:,'num_songs'] = plylst_use['songs_id'].map(len)
    plylst_use.loc[:,'num_tags'] = plylst_use['tags_id'].map(len)
    plylst_use = plylst_use.set_index('nid')
    
    plylst_train = plylst_use.iloc[:n_train,:]
    plylst_test = plylst_use.iloc[n_train:,:]
    
    # Make CSR
    
    row = np.repeat(range(n_train+n_test), plylst_use['num_songs'])
    col = [song for songs in plylst_use['songs_id'] for song in songs]
    dat = np.repeat(1, plylst_use['num_songs'].sum())
    ply_song_CSR = spr.csr_matrix((dat, (row, col)), shape=(n_train+n_test, n_songs))
    
    row = np.repeat(range(n_train+n_test), plylst_use['num_tags'])
    col = [tag for tags in plylst_use['tags_id'] for tag in tags]
    dat = np.repeat(1, plylst_use['num_tags'].sum())
    ply_tag_CSR = spr.csr_matrix((dat, (row, col)), shape=(n_train+n_test, n_tags))
    
    train_song_CSR = ply_song_CSR[:n_train]
    train_song_CSR = train_song_CSR.T
    
    ## CosSim
    
    norm = np.sqrt(train_song_CSR.multiply(train_song_CSR).sum(1))
    norm[norm==0]=1
    song_Cos = train_song_CSR.multiply(1/norm)
    song_Cos = song_Cos.dot(song_Cos.T)
    
    train_tag_CSR = ply_tag_CSR[:n_train]
    train_tag_CSR = train_tag_CSR.T
    
    norm = np.sqrt(train_tag_CSR.multiply(train_tag_CSR).sum(1))
    norm[norm==0]=1
    tag_Cos = train_tag_CSR.multiply(1/norm)
    tag_Cos = tag_Cos.dot(tag_Cos.T)
    
    # Split data
    
    plylst_train = plylst_use.iloc[:n_train,:]
    plylst_test = plylst_use.iloc[n_train:,:]
    
    test = plylst_test
    
    # extract candidates
    ## extrack at most 500 songs and 50 tags
    
    def rec(pids):
    
      res = []
    
      for pid in tqdm(pids):
        p = np.asarray(ply_song_CSR[pid].todense())[0]
    
        songs_already = test.loc[pid, "songs_id"]
    
        cand_song = song_Cos.dot(p)    
        cand_song_idx = cand_song.reshape(-1).argsort()[-600:][::-1]
        cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:500]
        
        rec_song_idx = [song_sid_id[i] for i in cand_song_idx]
        rec_song_value = cand_song[cand_song_idx].tolist()
        
        
        q = np.asarray(ply_tag_CSR[pid].todense())[0]
        
        tags_already = test.loc[pid, "tags_id"]
    
        cand_tag = tag_Cos.dot(q)
        cand_tag_idx = cand_tag.reshape(-1).argsort()[-60:][::-1]
        cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:50]
        
        
        rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]
        rec_tag_value = cand_tag[cand_tag_idx].tolist()
    
        res.append({
                    "id": plylst_nid_id[pid],
                    "songs": rec_song_idx,
                    "tags": rec_tag_idx,
                    "songs_score" : rec_song_value,
                    "tags_score" : rec_tag_value
                })
    
      return res
  
    answers = rec(test.index)
    
    # Standardizing Score
    
    for i in range(len(answers)):
    
        song_arr = np.array(answers[i]['songs_score'])
        if np.std(song_arr) != 0 :
            mean = np.mean(song_arr)
            std = np.std(song_arr)
            answers[i]['songs_score'] = ((song_arr - mean) / std).tolist()
        else:
            answers[i]['songs_score'] = answers[i]['songs_score']
    
    
        tag_arr = np.array(answers[i]['tags_score'])
        if np.std(tag_arr) != 0 :
            mean = np.mean(tag_arr)
            std = np.std(tag_arr)
            answers[i]['tags_score'] = ((song_arr - mean) / std).tolist()
        else:
            answers[i]['tags_score'] = answers[i]['tags_score']
            
    return answers