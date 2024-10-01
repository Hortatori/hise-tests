from HISEvent import hier_2D_SE_mini, get_global_edges, search_stable_points
from utils import evaluate, decode
from datetime import datetime
import math
import numpy as np
import pickle
import pandas as pd
import os
from os.path import exists
import time
import logging
from sklearn.metrics import f1_score

def get_stable_point(path):
    stable_point_path = path + 'stable_point.pkl'
    if not exists(stable_point_path):
        embeddings_path = path + 'SBERT_embeddings.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print("before search stable point")
        first_stable_point, global_stable_point = search_stable_points(embeddings)
        stable_points = {'first': first_stable_point, 'global': global_stable_point}
        print("succeeded search stable point")
        with open(stable_point_path, 'wb') as fp:
            pickle.dump(stable_points, fp)
        print('stable points stored.')

    with open(stable_point_path, 'rb') as f:
        stable_points = pickle.load(f)
    print('stable points loaded.')
    return stable_points

def cluster_event_match(data, pred):
    print(f"size labels list {len(data)}, size pred list {len(pred)}")
    df = pd.DataFrame(data, columns = ["label"])
    df["pred"] = pd.Series(pred, dtype=df.label.dtype)
    df = df[df.label.notna()]
    print("label", len(df.label.unique()))
    print("pred", len(df.pred.unique()))
    t0 = time.time()

    match = df.groupby(["label", "pred"], sort=False).size().reset_index(name="a")
    b, c = [], []
    for idx, row in match.iterrows():
        b_ = ((df["label"] != row["label"]) & (df["pred"] == row["pred"]))
        b.append(b_.sum())
        c_ = ((df["label"] == row["label"]) & (df["pred"] != row["pred"]))
        c.append(c_.sum())
    logging.info("match clusters with events took: {} seconds".format(time.time() - t0))
    match["b"] = pd.Series(b)
    match["c"] = pd.Series(c)
    # recall = nb true positive / (nb true positive + nb false negative)
    match["r"] = match["a"] / (match["a"] + match["c"])
    # precision = nb true positive / (nb true positive + nb false positive)
    match["p"] = match["a"] / (match["a"] + match["b"])
    match["f1"] = 2 * match["r"] * match["p"] / (match["r"] + match["p"])
    match = match.sort_values("f1", ascending=False)
    macro_average_f1 = match.drop_duplicates("label").f1.mean()
    macro_average_precision = match.drop_duplicates("label").p.mean()
    macro_average_recall = match.drop_duplicates("label").r.mean()
    logging.info("macro_average_precision {}, macro_average_recall {}, macro_average_f1 {}".format(macro_average_precision, macro_average_recall, macro_average_f1))
    print(f"macro_average_precision {macro_average_precision}, macro_average_recall {macro_average_recall}, macro_average_f1 {macro_average_f1}")
    return macro_average_precision, macro_average_recall, macro_average_f1

def run_hier_2D_SE_mini_Event2012_open_set(n = 400, e_a = True, e_s = True, test_with_one_block = True):
    save_path = './data/Event2012/open_set/'
    plist = []
    rlist = []
    f1list = []
    if test_with_one_block:
        blocks = [20]
    else:
        blocks = [i+1 for i in range(21)]
    for block in blocks:
        print('\n\n====================================================')
        print('block: ', block)
        print(datetime.now().strftime("%H:%M:%S"))

        folder = f'{save_path}{block}/'
        
        # load message embeddings
        embeddings_path = folder + 'SBERT_embeddings.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print("embedding shape ", embeddings.shape)
        df_np = np.load(f'{folder}{block}.npy', allow_pickle=True)
        df = pd.DataFrame(data=df_np, columns=["original_index", "event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
                "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities", 
                "words", "filtered_words", "sampled_words", "date"])
        all_node_features = [[str(u)] + \
            [str(each) for each in um] + \
            [h.lower() for h in hs] + \
            e \
            for u, um, hs, e in \
            zip(df['user_id'], df['user_mentions'],  df['hashtags'], df['entities'])]
        
        stable_points = get_stable_point(folder)
        if e_a == False: # only rely on e_s (semantic-similarity-based edges)
            default_num_neighbors = stable_points['global']
        else:
            default_num_neighbors = stable_points['first']
        if default_num_neighbors == 0: 
            default_num_neighbors = math.ceil((len(embeddings)/1000)*10)
        
        global_edges = get_global_edges(all_node_features, embeddings, default_num_neighbors, e_a = e_a, e_s = e_s)

        corr_matrix = np.corrcoef(embeddings)
        np.fill_diagonal(corr_matrix, 0)
        weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0]-1, edge[1]-1]) for edge in global_edges \
            if corr_matrix[edge[0]-1, edge[1]-1] > 0] # node encoding starts from 1
        
        division = hier_2D_SE_mini(weighted_global_edges, len(embeddings), n = n)
        print(datetime.now().strftime("%H:%M:%S"))
        prediction = decode(division)

        labels_true = df['event_id'].tolist()
        n_clusters = len(list(set(labels_true)))
        print('n_clusters gt: ', n_clusters)
        p, r, f1 = cluster_event_match(labels_true, prediction)
        plist.append(p)
        rlist.append(r)
        f1list .append(f1)
        nmi, ami, ari = evaluate(labels_true, prediction)
        print('n_clusters pred: ', len(division))
        print('nmi: ', nmi)
        print('ami: ', ami)
        print('ari: ', ari)
        # f1 = f1_score(labels_true, prediction, average = "macro")
        # print("f1 ", f1)
        # macro_average_precision, macro_average_recall, macro_average_f1 = cluster_event_match(labels_true,prediction)
        # logging.info("macro_average_precision {}, macro_average_recall {}, macro_average_f1 {}".format(macro_average_precision, macro_average_recall, macro_average_f1))
    print(f"p score mean : {sum(plist)/len(plist)}")
    print(f"r score mean : {sum(rlist)/len(rlist)}")
    print(f"f1 score : {sum(f1list)/len(f1list)}")
    return

def run_hier_2D_SE_mini_Event2012_closed_set(n = 300, e_a = True, e_s = True):
    save_path = './data/Event2012/closed_set/'

    #load test_set_df
    # test_set_df_np_path = save_path + 'test_set.npy'
    test_set_df_np_path = save_path + 'all_set.npy'
    test_df_np = np.load(test_set_df_np_path, allow_pickle=True)
    test_df = pd.DataFrame(data=test_df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
            "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities", 
            "words", "filtered_words", "sampled_words"])
    print("Dataframe loaded.")
    all_node_features = [[str(u)] + \
        [str(each) for each in um] + \
        [h.lower() for h in hs] + \
        e \
        for u, um, hs, e in \
        zip(test_df['user_id'], test_df['user_mentions'],  test_df['hashtags'], test_df['entities'])]

    # load embeddings of the test set messages
    print("loading Sbert embeddings")
    with open(f'{save_path}/SBERT_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    print("embeddings loaded")

    stable_points = get_stable_point(save_path)
    print("ended stable point")
    default_num_neighbors = stable_points['first']

    global_edges = get_global_edges(all_node_features, embeddings, default_num_neighbors, e_a = e_a, e_s = e_s)
    print("ended global edges")
    corr_matrix = np.corrcoef(embeddings)
    np.fill_diagonal(corr_matrix, 0)
    weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0]-1, edge[1]-1]) for edge in global_edges \
        if corr_matrix[edge[0]-1, edge[1]-1] > 0] # node encoding starts from 1
    print("running 2d se")
    division = hier_2D_SE_mini(weighted_global_edges, len(embeddings), n = n)
    prediction = decode(division)

    labels_true = test_df['event_id'].tolist()
    n_clusters = len(list(set(labels_true)))
    print('n_clusters gt: ', n_clusters)
    print("TEST F1 HERE")
    p, r, f1 = cluster_event_match(labels_true, prediction)
    nmi, ami, ari = evaluate(labels_true, prediction)
    print('n_clusters pred: ', len(division))
    print('nmi: ', nmi)
    print('ami: ', ami)
    print('ari: ', ari)
    return

def run_hier_2D_SE_mini_Event2018_open_set(n = 300, e_a = True, e_s = True, test_with_one_block = True):
    save_path = './data/Event2018/open_set/'
    plist = []
    rlist = []
    f1list = []
    if test_with_one_block:
        blocks = [16]
    else:
        blocks = [i+1 for i in range(16)]
    for block in blocks:
        print('\n\n====================================================')
        print('block: ', block)
        print(datetime.now().strftime("%H:%M:%S"))

        folder = f'{save_path}{block}/'
        
        # load message embeddings
        embeddings_path = folder + 'SBERT_embeddings.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        df_np = np.load(f'{folder}{block}.npy', allow_pickle=True)
        df = pd.DataFrame(data=df_np, columns=["original_index", "tweet_id", "user_name", "text", "time", "event_id", "user_mentions", \
                "hashtags", "urls", "words", "created_at", "filtered_words", "entities", "sampled_words", "date"])
        all_node_features = [list(set([str(u)] + \
            [str(each) for each in um] + \
            [h.lower() for h in hs] + \
            e)) \
            for u, um, hs, e in \
            zip(df['user_name'], df['user_mentions'],  df['hashtags'], df['entities'])]
        
        stable_points = get_stable_point(folder)
        if e_a == False: # only rely on e_s (semantic-similarity-based edges)
            default_num_neighbors = stable_points['global']
        else:
            default_num_neighbors = stable_points['first']
        if default_num_neighbors == 0: 
            default_num_neighbors = math.ceil((len(embeddings)/1000)*10)
        
        global_edges = get_global_edges(all_node_features, embeddings, default_num_neighbors, e_a = e_a, e_s = e_s)

        corr_matrix = np.corrcoef(embeddings)
        np.fill_diagonal(corr_matrix, 0)
        weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0]-1, edge[1]-1]) for edge in global_edges \
            if corr_matrix[edge[0]-1, edge[1]-1] > 0] # node encoding starts from 1
        
        division = hier_2D_SE_mini(weighted_global_edges, len(embeddings), n = n)
        print(datetime.now().strftime("%H:%M:%S"))

        prediction = decode(division)

        labels_true = df['event_id'].tolist()
        n_clusters = len(list(set(labels_true)))
        print('n_clusters gt: ', n_clusters)
        print("TEST F1 HERE")
        p, r, f1 = cluster_event_match(labels_true, prediction)
        plist.append(p)
        rlist.append(r)
        f1list .append(f1)

        nmi, ami, ari = evaluate(labels_true, prediction)
        print('n_clusters pred: ', len(division))
        print('nmi: ', nmi)
        print('ami: ', ami)
        print('ari: ', ari)
    
    print(f"p score mean : {sum(plist)/len(plist)}")
    print(f"r score mean : {sum(rlist)/len(rlist)}")
    print(f"f1 score : {sum(f1list)/len(f1list)}")
    
        
    return

def run_hier_2D_SE_mini_Event2018_closed_set(n = 800, e_a = True, e_s = True):
    save_path = './data/Event2018/closed_set/'

    #load test_set_df
    #test_set_df_np_path = save_path + 'test_set.npy'
    test_set_df_np_path = save_path + 'all_set.npy'
    test_df_np = np.load(test_set_df_np_path, allow_pickle=True)
    test_df = pd.DataFrame(data=test_df_np, columns=["tweet_id", "user_name", "text", "time", "event_id", "user_mentions", \
            "hashtags", "urls", "words", "created_at", "filtered_words", "entities", "sampled_words"])
    print("Dataframe loaded.")
    all_node_features = [list(set([str(u)] + \
        [str(each) for each in um] + \
        [h.lower() for h in hs] + \
        e)) \
        for u, um, hs, e in \
        zip(test_df['user_name'], test_df['user_mentions'],  test_df['hashtags'], test_df['entities'])]

    # load embeddings of the test set messages
    with open(f'{save_path}/SBERT_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    stable_points = get_stable_point(save_path)
    default_num_neighbors = stable_points['first']

    global_edges = get_global_edges(all_node_features, embeddings, default_num_neighbors, e_a = e_a, e_s = e_s)
    corr_matrix = np.corrcoef(embeddings)
    np.fill_diagonal(corr_matrix, 0)
    weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0]-1, edge[1]-1]) for edge in global_edges \
        if corr_matrix[edge[0]-1, edge[1]-1] > 0] # node encoding starts from 1

    division = hier_2D_SE_mini(weighted_global_edges, len(embeddings), n = n)
    prediction = decode(division)

    labels_true = test_df['event_id'].tolist()
    n_clusters = len(list(set(labels_true)))
    print('n_clusters gt: ', n_clusters)
    print("TEST F1 HERE")
    p, r, f1 = cluster_event_match(labels_true, prediction)
    nmi, ami, ari = evaluate(labels_true, prediction)
    print('n_clusters pred: ', len(division))
    print('nmi: ', nmi)
    print('ami: ', ami)
    print('ari: ', ari)
    return

if __name__ == "__main__":
    # to run all message blocks, set test_with_one_block to False
    # run_hier_2D_SE_mini_Event2012_open_set(n = 400, e_a = True, e_s = True, test_with_one_block = True)
    run_hier_2D_SE_mini_Event2012_closed_set(n = 300, e_a = True, e_s = True)
    # run_hier_2D_SE_mini_Event2018_open_set(n = 300, e_a = True, e_s = True, test_with_one_block = False)
    # run_hier_2D_SE_mini_Event2018_closed_set(n = 800, e_a = True, e_s = True)
    
