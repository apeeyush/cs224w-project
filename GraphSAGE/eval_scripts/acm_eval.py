from __future__ import print_function
import json
import numpy as np

from networkx.readwrite import json_graph
from argparse import ArgumentParser

def get_class_labels(G, ids):
    class_map = {}
    # Compute class map
    for node in G.nodes():
        class_map[node] = G.node[node]['publicationYear']
    classes = [class_map[i] for i in ids]
    return classes

def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    np.random.seed(1)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(train_embeds, train_labels)
    print(model.predict(test_embeds))
    score = model.score(test_embeds, test_labels)
    print("Score: {}".format(score))
    # from sklearn.linear_model import SGDClassifier
    # from sklearn.dummy import DummyClassifier
    # from sklearn.metrics import f1_score
    # dummy = DummyClassifier()
    # dummy.fit(train_embeds, train_labels)
    # log = SGDClassifier(loss="log", n_jobs=10)
    # log.fit(train_embeds, train_labels)
    # print("F1 score:", f1_score(test_labels, log.predict(test_embeds), average="micro"))
    # print("Random baseline f1 score:", f1_score(test_labels, dummy.predict(test_embeds), average="micro"))

if __name__ == '__main__':
    # parser = ArgumentParser("Run evaluation on citation data.")
    # parser.add_argument("dataset_dir", help="Path to directory containing the dataset.")
    # parser.add_argument("embed_dir", help="Path to directory containing the learned node embeddings.")
    # parser.add_argument("setting", help="Either val or test.")
    # args = parser.parse_args()
    dataset_dir = "../"  # args.dataset_dir
    data_dir = "feat" # "unsup-../graphsage_mean_small_0.000010"  or "feat"
    setting = "test"  # args.setting

    print("Loading data...")
    G = json_graph.node_link_graph(json.load(open(dataset_dir + "/acm-G.json")))
     
    train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids = [n for n in G.nodes() if G.node[n][setting]]
    test_labels = get_class_labels(G, test_ids)
    train_labels = get_class_labels(G, train_ids)

    if data_dir == "feat":
        print("Using only features..")
        feats = np.load(dataset_dir + "/acm-feats.npy")
        feat_id_map = json.load(open(dataset_dir + "/acm-id_map.json"))
        feat_id_map = {int(id):val for id,val in feat_id_map.iteritems()}
        train_feats = feats[[feat_id_map[id] for id in train_ids]] 
        test_feats = feats[[feat_id_map[id] for id in test_ids]] 
        print("Running regression..")
        run_regression(train_feats, train_labels, test_feats, test_labels)

    elif "n2v" in data_dir:
        print("Using n2v vectors.")
        base_embeds = np.load(data_dir + "/val.npy")
        base_id_map = {}
        with open(data_dir + "/val.txt") as fp:
            for i, line in enumerate(fp):
                base_id_map[int(line.strip())] = i
        tuned_embeds = np.load(data_dir + "/val-test.npy")
        tuned_id_map = {}
        with open(data_dir + "/val-test.txt") as fp:
            for i, line in enumerate(fp):
                tuned_id_map[int(line.strip())] = i
        train_embeds = base_embeds[[base_id_map[id] for id in train_ids]] 
        test_embeds = tuned_embeds[[tuned_id_map[id] for id in test_ids]] 

        print("Running regression..")
        run_regression(train_embeds, train_labels, test_embeds, test_labels)

        # loading feats
        feats = np.load(dataset_dir + "/isi-feats.npy")
        feat_id_map = json.load(open(dataset_dir + "/isi-id_map.json"))
        feat_id_map = {int(id):val for id,val in feat_id_map.iteritems()}
        train_feats = feats[[feat_id_map[id] for id in train_ids]] 
        test_feats = feats[[feat_id_map[id] for id in test_ids]] 
        train_embeds = np.hstack([train_feats, train_embeds])
        test_embeds = np.hstack([test_feats, test_embeds])
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(train_embeds)
        train_embeds = scaler.transform(train_embeds)
        test_embeds = scaler.transform(test_embeds)

        print("Running regression with feats..")
        run_regression(train_embeds, train_labels, test_embeds, test_labels)
    else:
        embeds = np.load(data_dir + "/val.npy")
        id_map = {}
        with open(data_dir + "/val.txt") as fp:
            for i, line in enumerate(fp):
                id_map[int(line.strip())] = i
        train_embeds = embeds[[id_map[id] for id in train_ids]] 
        test_embeds = embeds[[id_map[id] for id in test_ids]] 

        print("Running regression..")
        run_regression(train_embeds, train_labels, test_embeds, test_labels)
