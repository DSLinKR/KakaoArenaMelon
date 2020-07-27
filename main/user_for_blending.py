from collections import Counter

import numpy as np
import pandas as pd

import scipy.sparse as spr
from tqdm import tqdm
from math import sqrt


def item_for_blening(train, test):
    
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
    
    
    # Make CSR
    
    row = np.repeat(range(n_train+n_test), plylst_use['num_songs'])
    col = [song for songs in plylst_use['songs_id'] for song in songs]
    dat = np.repeat(1, plylst_use['num_songs'].sum())
    all_song = spr.csr_matrix((dat, (row, col)), shape=(n_train+n_test, n_songs))
    

    row = np.repeat(range(n_train+n_test), plylst_use['num_tags'])
    col = [tag for tags in plylst_use['tags_id'] for tag in tags]
    dat = np.repeat(1, plylst_use['num_tags'].sum())
    all_tag = spr.csr_matrix((dat, (row, col)), shape=(n_train+n_test, n_tags))

    
    ## CosSim
    
    norm = np.sqrt(all_song.multiply(all_song).sum(1))
    norm[norm==0]=1
    ply_song = all_song.multiply(1/norm).tocsr()
    train_song = ply_song[:n_train]
    
    norm = np.sqrt(all_tag.multiply(all_tag).sum(1))
    norm[norm==0]=1
    ply_tag = all_tag.multiply(1/norm).tocsr()
    train_tag = ply_tag[:n_train]
        

    # IDF

    m = np.ones(all_song.shape[0])
    song_frequency = all_song.T.dot(m)


    desired_length = all_song.shape[1]
    idf_vector = np.zeros(desired_length)

    for i in tqdm(range(desired_length)):
        idf_vector[i] = 1/(sqrt(song_frequency[i]-1)+1)
    

    # Split data
    
    plylst_train = plylst_use.iloc[:n_train,:]
    plylst_test = plylst_use.iloc[n_train:,:]
    
    test = plylst_test
    
    
    # CSR for rec songs
    
    row = np.repeat(range(n_train), plylst_train['num_songs'])
    col = [song for songs in plylst_train['songs_id'] for song in songs]
    dat = np.repeat(1, plylst_train['num_songs'].sum())
    train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_songs))

    row = np.repeat(range(n_train), plylst_train['num_tags'])
    col = [tag for tags in plylst_train['tags_id'] for tag in tags]
    dat = np.repeat(1, plylst_train['num_tags'].sum())
    train_tags_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_tags))
    
    idf_vector = sparse.csr_matrix(idf_vector)

    M_train_songs = train_songs_A.multiply(idf_vector)
    M_train_songs_T = M_train_songs.T
    train_songs_A_T = train_songs_A.T
    train_tags_A_T = train_tags_A.T
    
    # extract candidates
    ## extrack at most 500 songs and 50 tags
    
    # pids = test.index

    def norm_score(pids):
        res = []
        for pid in tqdm(pids):
            p = np.asarray(ply_song[pid].todense())[0]

            val = train_song.dot(p).reshape(-1)
            
            songs_already = test.loc[pid, "songs_id"]
            tags_already = test.loc[pid, "tags_id"]
 
            cand_song = M_train_songs_T.dot(val)
            cand_song_idx = cand_song.reshape(-1).argsort()[-600:][::-1]
            cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:500]
            rec_song_idx = [song_sid_id[i] for i in cand_song_idx]
            rec_song_value = cand_song[cand_song_idx]
        
            if np.std(rec_song_value)!=0:
                song_value_mean = np.mean(rec_song_value)
                song_value_sd = np.std(rec_song_value)
                normalized_songs = (rec_song_value - song_value_mean)/song_value_sd
            else: normalized_songs = rec_song_value
        
        
            cand_tag = train_tags_A_T.dot(val)
            cand_tag_idx = cand_tag.reshape(-1).argsort()[-100:][::-1]
            cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:50]
            rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]
            rec_tag_value = cand_tag[cand_tag_idx]
        
            if np.std(rec_tag_value)!=0:
                tag_value_mean = np.mean(rec_tag_value)
                tag_value_sd = np.std(rec_tag_value)
                normalized_tags = (rec_tag_value - tag_value_mean)/tag_value_sd
            else: normalized_tags = rec_tag_value
        
        # index들을 위에서처럼 plylst_nid_id[pid], rec_song_idx, rec_tag_idx로 index를 줘야 하나??
            res.append({
                        "id": plylst_nid_id[pid],
                        "songs": rec_song_idx,
                        "tags": rec_tag_idx,
                        "normalized song score": normalized_songs,
                        "normalized tag score": normalized_tags
                        })

        return res
    
    nm_score = norm_score(test.index)
    
    for i in range(len(nm_score)):
        nm_score[i]['normalized song score'] = nm_score[i]['normalized song score'].tolist()
        nm_score[i]['normalized tag score'] = nm_score[i]['normalized tag score'].tolist()
        
        
    return nm_score