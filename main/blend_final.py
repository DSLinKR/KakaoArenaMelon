# Cold Start 포함한 결과들 삭제
import pandas as pd
import numpy as np
from collections import Counter

# Modules
from als_for_blending import als_for_blending





def blending_final(train_set, test_set) :
    tr = train_set
    te = test_set

    # Cold Start
    no_songs = te[te.songs.apply(len)==0]
    cold_start = no_songs[no_songs.tags.apply(len)==0]
    cs_id = cold_start.id

    # index of rows that are not cold start
    n_cs_id = [x for x in range(len(te)) if x not in cs_id.index]

    # Candidates
    can_als = pd.DataFrame(als_for_blending(tr,te))
    can_item = pd.DataFrame(item_for_blending(tr,te))
    can_user = pd.DataFrame(user_for_blending(tr,te))

    # only warm(?) starts
    can_item = can_item.iloc[n_cs_id,:]
    can_user = can_user.iloc[n_cs_id, :]

    # reset index
    can_item = can_item.reset_index(drop=True)
    can_user = can_user.reset_index(drop=True)


    # changing als values(다른 파일들과의 통일성 위해 score row의 각 딕셔너리에서 value 값만 뽑아냄)
    def get_values(dictionary) :
        return list( dictionary.values() )
        
    can_als = can_als.assign(songs_score = can_als.songs_score.apply(get_values))
    can_als = can_als.assign(tags_score = can_als.tags_score.apply(get_values))


    #  Counter 함수에 문제가 없도록, 모든 값을 양수로 바꾸기 위해 큰 값을 더해줌

    def plus_ten(ls) :
        return ( np.array(ls) + 10 ).tolist()

    can_als = can_als.assign(songs_score = can_als.songs_score.apply(plus_ten) )
    can_user = can_user.assign(songs_score = can_user['normalized song score'].apply(plus_ten) )
    can_item = can_item.assign(songs_score = can_item.songs_score.apply(plus_ten) )

    can_als = can_als.assign(tags_score = can_als.tags_score.apply(plus_ten) )
    can_user = can_user.assign(tags_score = can_user['normalized tag score'].apply(plus_ten) )
    can_item = can_item.assign(tags_score = can_item.tags_score.apply(plus_ten) )


    # Weighting
    # Songs
    ######## change here #######
    w_als = 0.2
    w_user = 0.3
    w_item = 0.3
    ##########################

    def als_apply_weight(ls):
        return (np.array(ls) * w_als).tolist()
    
    def user_apply_weight(ls):
        return (np.array(ls) * w_user).tolist()
        
        
    def item_apply_weight(ls):
        return (np.array(ls) * w_item).tolist()

    can_als = can_als.assign(songs_score = can_als.songs_score.apply(als_apply_weight))
    can_user = can_user.assign(songs_score = can_user.songs_score.apply(user_apply_weight))
    can_item = can_item.assign(songs_score = can_item.songs_score.apply(item_apply_weight))

    # Tags
    ######## change here #######
    w_als_t = 0.1
    w_user_t = 0.3
    w_item_t = 0.1
    ##########################

    def als_apply_weight_t(ls):
        return (np.array(ls) * w_als_t).tolist()
    
    def user_apply_weight_t(ls):
        return (np.array(ls) * w_user_t).tolist()
        
        
    def item_apply_weight_t(ls):
        return (np.array(ls) * w_item_t).tolist()

    can_als = can_als.assign(tags_score = can_als.tags_score.apply(als_apply_weight_t))
    can_user = can_user.assign(tags_score = can_user.tags_score.apply(user_apply_weight_t))
    can_item = can_item.assign(tags_score = can_item.tags_score.apply(item_apply_weight_t))


    # Song Blending
    def score_map(item_score_list) :
        item_dict = {}
        item_len = int(len(item_score_list) / 2)
        for item, score in zip(item_score_list[:item_len], item_score_list[item_len:]) :
            item_dict[str(item)] = score
        return item_dict

    # make a temporary list that contains both items and scores
    can_user = can_user.assign(temp = can_user.songs + can_user['normalized song score'])
    can_item = can_item.assign(temp = can_item.songs + can_item.songs_score)
    can_als = can_als.assign(temp = can_als.songs + can_als.songs_score)

    # update the score column with dictionaries
    can_user = can_user.assign(songs_score = can_user.temp.apply(score_map) )
    can_item = can_item.assign(songs_score = can_item.temp.apply(score_map) )
    can_als = can_als.assign(songs_score= can_als.temp.apply(score_map))

    # make dataframe for summing
    df_sum = pd.DataFrame()

    df_sum['als_score'] = can_als.songs_score
    df_sum['user_score'] = can_user.songs_score
    df_sum['item_score'] = can_item.songs_score

    # change to counter objects and then sum

    df_sum = df_sum.applymap(Counter)
    df_sum = df_sum.assign(sum_score = df_sum.als_score + df_sum.user_score + df_sum.item_score)

    # sort based on values
    def sort_dictionary(dictionary) :
        dict_sorted_by_values ={k : v for k, v in sorted(dictionary.items(), key=lambda x: x[1], reverse=True)}
        return dict_sorted_by_values

    df_sum = df_sum.assign(sum_score = df_sum.sum_score.apply(sort_dictionary))

    def get_high_songs(dictionary) :
        top_500_items = list(dictionary.keys() )[:500]
        return top_500_items
        
    top_songs = df_sum.sum_score.apply(get_high_songs)

    # Tag Blending
    # make a temporary list that contains both items and scores
    can_als = can_als.assign(temp2 = can_als.tags + can_als.tags_score)
    can_user = can_user.assign(temp2 = can_user.tags + can_user['normalized tag score'] )
    can_item = can_item.assign(temp2 = can_item.tags + can_item.tags_score)

    # update the score column with dictionaries
    can_als = can_als.assign(tags_score = can_als.temp2.apply(score_map))
    can_user = can_user.assign(tags_score = can_user.temp2.apply(score_map) )
    can_item = can_item.assign(tags_score = can_item.temp2.apply(score_map) )

    # make dataframe for summing
    df_sum2 = pd.DataFrame()

    df_sum2['als_score'] = can_als.tags_score
    df_sum2['user_score'] = can_user.tags_score
    df_sum2['item_score'] = can_item.tags_score

    # change to counter objects and then sum
    df_sum2 = df_sum2.applymap(Counter)
    df_sum2 = df_sum2.assign(sum_score = df_sum2.als_score + df_sum2.user_score + df_sum2.item_score)


    df_sum2 = df_sum2.assign(sum_score = df_sum2.sum_score.apply(sort_dictionary))

    def get_high_tags(dictionary) :
        top_50_items = list(dictionary.keys() )[:50]
        return top_50_items
        
    top_tags = df_sum2.sum_score.apply(get_high_tags)

    returnval = []

    for _id, top_song, top_tag in zip(can_als.id, top_songs, top_tags): # cold start 포함 안 한 부분
        returnval.append({
            "id": _id,
            "songs": list(map(int,top_song)),
            "tags": top_tag
        })

    return returnval
