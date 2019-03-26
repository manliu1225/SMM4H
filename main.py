import argparse
import json
import os
import re
import time
from multiprocessing import Pool

from tqdm import tqdm

import data
# import feature_context
import feature_namelist
import feature_ascii
import feature_wordvec
import feature_brown

def extract_from_instance(instance_idx: int, features_list: dict, debug=False):

    instance_tokens, instance_pos, instance_bio = token_pos_bio[instance_idx]
    untokenized = "".join(instance_tokens)
    instance_tokens_lowered = [token.lower() for token in instance_tokens]

    # Basic features
    word_idx, pos_idx, ngram_0_1 = [], [], []

    # WordVeCluster features
    wordvec_cluster, wordvec_cluster2, wordvec_cluster3, wordvec_cluster4, wordvec_cluster5 = [], [], [], [], []

    # BrownCluster feature
    brown_cluster = []

    # Namelist features
    namelist_APP, namelist_GAME, namelist_O = [], [], []
    namelist_idx = feature_namelist.get_namelist_match_idx(
            features_dict["nameListFeature"], instance_tokens_lowered) 

    # Namelist2 features
    namelist2_APP, namelist2_GAME = [], []
    namelist2_idx = feature_namelist.get_namelist_match_idx(
        features_dict["nameList2Feature"], instance_tokens_lowered) 


    # # AllAscii features
    all_ascii = []
    first_ascii = []

    for idx, token in enumerate(instance_tokens):

        word_idx.append(features_dict["Word"].get("Word:"+token, "_"))
        pos_idx.append(features_dict["Pos"].get("Pos:"+instance_pos[idx], "_"))
        ngram_0_1.append(features_dict["CharacterNgram_0_1"].get("CharacterNgram_0_1:"+token[0:1], "_"))

        wordvec_cluster.append(feature_wordvec.get_wordvec_cluster_value(
            token, features_dict, "wordVecClusterFeatureFileList", "WordVecCluster"))
        wordvec_cluster2.append(feature_wordvec.get_wordvec_cluster_value(
            token, features_dict, "wordVecClusterFeature2FileList", "WordVecCluster2"))
        wordvec_cluster3.append(feature_wordvec.get_wordvec_cluster_value(
            token, features_dict, "wordVecClusterFeature3FileList", "WordVecCluster3"))
        wordvec_cluster4.append(feature_wordvec.get_wordvec_cluster_value(
            token, features_dict, "wordVecClusterFeature4FileList", "WordVecCluster4"))
        wordvec_cluster5.append(feature_wordvec.get_wordvec_cluster_value(
            token, features_dict, "wordVecClusterFeature5FileList", "WordVecCluster5"))

        brown_cluster.append(feature_brown.get_brown_cluster_value(
            token, features_dict, "BrownClusterFile", "BrownCluster", prefixLength = 14, minCount = 1))
        
        all_ascii.append(feature_ascii.get_allascii_value(token, features_dict, "AllAscii"))
        first_ascii.append(feature_ascii.get_firstascii_value(token, idx, features_dict, "FirstAscii"))

        if idx in namelist2_idx:
            start = ""
            # start = "_1" if namelist2_idx.get(idx, "")["pos"] == 0 else ""
            namelist2_dict = features_dict["NameList2"]
            namelist2_APP.append(namelist2_dict["NameList2:APP"] if "@@@APP" in namelist2_idx.get(idx, "")["labels"] else "_")
            namelist2_GAME.append(namelist2_dict["NameList2:GAME"] if "@@@GAME" in namelist2_idx.get(idx, "")["labels"] else "_")
        else:
            namelist2_APP.append("_")
            namelist2_GAME.append("_")

        if idx in namelist_idx:
            namelist_dict = features_dict["NameList"]
            # print(namelist_idx.get(idx, "")["pos"])
            # start = ""
            start = "_1" if namelist_idx.get(idx, "")["pos"] == 0 else ""
            namelist_APP.append(namelist_dict["NameList:APP"] if "APP" in namelist_idx.get(idx, "")["labels"] else "_")
            namelist_GAME.append(namelist_dict["NameList:GAME"] if "GAME" in namelist_idx.get(idx, "")["labels"] else "_")
            namelist_O.append(namelist_dict["NameList:O"] if "O" in namelist_idx.get(idx, "")["labels"] else "_")
        else:
            namelist_APP.append("_")
            namelist_GAME.append("_")
            namelist_O.append("_")


    def print_debug_result():
        def print_result(x, y, feature_name):
            same_pass = lambda x,y: "PASS" if [str(x_i) for x_i in x] == [str(y_i) for y_i in y] else "FAIL"
            same_result = same_pass(x, y)
            if same_result == "FAIL":
                print("{}: {}".format(feature_name, same_result))
                print("\n{}".format([(instance_tokens[idx], str(label)) for idx, label in enumerate(x)]))
                print("{}\n".format([(instance_tokens[idx], str(label)) for idx, label in enumerate(y)]))

        print("== Running matching test {} ===".format(instance_idx))
        # print("Instance {}: {}".format(instance_idx, token_pos_bio[instance_idx]))

        print_result(target_features[instance_idx][0], word_idx, feature_name="word_idx")
        print_result(target_features[instance_idx][1], pos_idx, feature_name="pos_idx")

        print_result(target_features[instance_idx][2], wordvec_cluster, feature_name="wordvec_cluster")
        print_result(target_features[instance_idx][3], wordvec_cluster2, feature_name="wordvec_cluster2")
        print_result(target_features[instance_idx][4], wordvec_cluster3, feature_name="wordvec_cluster3")
        print_result(target_features[instance_idx][5], wordvec_cluster4, feature_name="wordvec_cluster4")
        print_result(target_features[instance_idx][6], wordvec_cluster5, feature_name="wordvec_cluster5")
        print_result(target_features[instance_idx][7], brown_cluster, feature_name="BrownCluster")
        print_result(target_features[instance_idx][8], ngram_0_1, feature_name="ngram_0_1")
        print_result(target_features[instance_idx][9], all_ascii, feature_name="all_ascii")
        print_result(target_features[instance_idx][10], first_ascii, feature_name="first_ascii")
        print_result(target_features[instance_idx][11], namelist2_APP, feature_name="namelist2_APP")
        print_result(target_features[instance_idx][12], namelist2_GAME, feature_name="namelist2_GAME")

        print_result(target_features[instance_idx][13], namelist_APP, feature_name="namelist_APP")
        print_result(target_features[instance_idx][14], namelist_GAME, feature_name="namelist_GAME")
        print_result(target_features[instance_idx][15], namelist_O, feature_name="namelist_O")
        print_result(target_features[instance_idx][16], instance_bio, feature_name="bio")

    if debug: print_debug_result()

    return [
        word_idx,
        pos_idx,
        wordvec_cluster,
        wordvec_cluster2,
        wordvec_cluster3,
        wordvec_cluster4,
        wordvec_cluster5,
        brown_cluster,
        ngram_0_1,
        all_ascii,
        first_ascii,
        namelist2_APP,
        namelist2_GAME,
        namelist_APP,
        namelist_GAME,
        namelist_O,
        instance_bio
    ]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Feature extraction for NER',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--debug', default=False, type=bool, help='DEBUG mode')
    parser.add_argument('--start_id', default=0, type=int, help='Where to start for debugging')
    parser.add_argument('--end_id', default=-1, type=int, help='Where to end for debugging')
    parser.add_argument('--threads', default=32, type=int, help='Number of threads')
    parser.add_argument('--input_files', default='dev.txt', type=str, help="Comma-seperated list of file")
    parser.add_argument('--output_dir', default=str(int(time.time())), type=str, help='Output folder name')
    args = parser.parse_args()

    BASE_DIR = "../source/"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    FEATURES_DIR = os.path.join(BASE_DIR, "features")
    MANUAL_MUSIC_DIR = os.path.join(BASE_DIR, "manual-mobile")

    config = data.load_config()

    print("Loading features...")
    features_filelist = os.listdir(FEATURES_DIR)
    features_dict = {}
    for feature_filename in features_filelist:
        feature_name = feature_filename[:-4]
        feature_filename = os.path.join(FEATURES_DIR, feature_filename)
        if not os.path.isfile(feature_filename): continue
        if feature_name == "RegexNamelist": continue
        features_dict[feature_name] = data.load_features_file(feature_filename)
    if config["features"]["WordVecCluster"]:
        features_dict["wordVecClusterFeatureFileList"] = data.load_features_file(
            os.path.join(MANUAL_MUSIC_DIR, config["resources"]["WordVecCluster"][0]), token_indexed=True)
    if config["features"]["WordVecCluster2"]:
        features_dict["wordVecClusterFeature2FileList"] = data.load_features_file(
            os.path.join(MANUAL_MUSIC_DIR, config["resources"]["WordVecCluster2"][0]), token_indexed=True)
    if config["features"]["WordVecCluster3"]:
        features_dict["wordVecClusterFeature3FileList"] = data.load_features_file(
            os.path.join(MANUAL_MUSIC_DIR, config["resources"]["WordVecCluster3"][0]), token_indexed=True)
    if config["features"]["WordVecCluster4"]:
        features_dict["wordVecClusterFeature4FileList"] = data.load_features_file(
            os.path.join(MANUAL_MUSIC_DIR, config["resources"]["WordVecCluster4"][0]), token_indexed=True)
    if config["features"]["WordVecCluster5"]:
        features_dict["wordVecClusterFeature5FileList"] = data.load_features_file(
            os.path.join(MANUAL_MUSIC_DIR, config["resources"]["WordVecCluster5"][0]), token_indexed=True)
    if config["features"]["NameList"]:
        features_dict["nameListFeature"] = feature_namelist.load_namelist(
            config["resources"]["NameList"], MANUAL_MUSIC_DIR, skip_first_row=True)
    if config["features"]["NameList2"]:
        features_dict["nameList2Feature"] = feature_namelist.load_namelist(
            config["resources"]["NameList2"], MANUAL_MUSIC_DIR, skip_first_row=False)
    if config["features"]["BrownCluster"]:
        features_dict["BrownClusterFile"] = data.load_features_file(
            os.path.join(MANUAL_MUSIC_DIR, config["resources"]["BrownCluster"][0]), token_indexed=True)
    print(features_dict.keys())

    print("Number of feature dict: {}".format(len(features_dict)))
    print("Features: {}".format(sorted(list(features_dict.keys()))))

    if not args.debug:

        # Setup runs folder
        timestamp = str(int(time.time()))
        save_folder = os.path.join("../runs", args.output_dir)

        def run(idx):
            return extract_from_instance(idx, config["features"])

        for filename in args.input_files.split(","):
            token_pos_bio = data.load_data(DATA_DIR, filename)
            print("start multiprocessing on {}...".format(filename[:-4]))
            result = []
            with Pool(processes=args.threads) as p:
                max_ = len(token_pos_bio)
                with tqdm(total=max_) as pbar:
                    for i, output in tqdm(enumerate(p.imap(run, range(max_)))):
                        result.append(output)
                        pbar.update()

            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            save_file = os.path.join(save_folder, filename[:-4])

            with open(save_file + "_raw", "w+") as f:
                for instance in token_pos_bio:
                    columns = len(instance)
                    rows = len(instance[0])
                    for row in range(rows):
                        f.write("\t".join([str(instance[col][row]) for col in range(columns)]) + "\n")
                    f.write("\n")

            with open(save_file, "w+") as f:
                for instance in result:
                    columns = len(instance)
                    rows = len(instance[0])
                    for row in range(rows):
                        f.write("\t".join([str(instance[col][row]) for col in range(columns)]) + "\n")
                    f.write("\n")

        # Save config file if feature extraction is completed
        with open(os.path.join(save_folder, "config.json"), "w+") as f:
            json.dump(config, f, indent=4,)

    else:
        token_pos_bio = data.load_data(DATA_DIR, "train_0.txt")
        target_features = data.load_target_file(os.path.join(BASE_DIR, "./java_samples/train.vector.txt"))
        print("Number of instances in target file: {}".format(len(target_features)))
        print("Number of features in an instance in a target file: {}".format(len(target_features[0])))
        print("First instance with feature {} at {}".format(
            9, data.get_first_instance_with_feature(target_features, 9, 0)))
        end_id = len(token_pos_bio) if args.end_id == -1 else args.end_id
        for idx in range(args.start_id, end_id):
            extract_from_instance(idx, config["features"], debug=True)
