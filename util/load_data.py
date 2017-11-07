from gensim.models import word2vec as Word2Vec
import pandas as pd
import numpy as np
import os
import sys
import copy
from util.Layer import *
import gzip
import random
import pickle

tf.set_random_seed(20170302)
np.random.seed(20170302)
random.seed(20170302)


class Data(object):

    def __init__(self, params, other_data=None, load_prob=False):
        self.embedding_dim = params["embedding_dim"]
        self.punctuations = {".": 1, ",": 1, ";": 1, "!": 1, "?": 1, "/": 1, """: 1, """: 1, "(": 1, ")": 1, "{": 1,
                             "}": 1, "[": 1, "]": 1, "=": 1, "\\": 1}
        if params["vector_src"] == "glove":
            vector_path = params["glove_file"]
        elif params["vector_src"] == "w2v":
            vector_path = params["w2v_file"]
        else:
            vector_path = ""
        self.stage = params["stage"]
        self.train_split = params["train_split"]
        self.test_split = params["test_split"]
        self.dev_split = params["dev_split"]
        if self.stage == "test":
            self.train_split = params["test_split"] # for seq len
        self.split_map = {}
        self.load_splits = ["train", "test", "dev"]
        if self.stage == "test":
            self.load_splits = [params["test_split"]]
            self.split_map[params["test_split"]] = "test"
        elif self.stage == "train":
            self.load_splits = [params["train_split"], params["dev_split"]]
            self.split_map[params["train_split"]] = "train"
            self.split_map[params["dev_split"]] = "dev"
            if load_prob:
                self.load_splits = [self.split_map["train"]]
        print "**"
        print self.split_map
        self.model_type = params["model_type"]
        self.data_format = params["data_format"]
        if (params["snli_pretrain"] or params["snli_train"]) and other_data is not None: # load SNLI data
            data_dir = params["snli_data_dir"]
            self.data_source = "snli"
            self.data_format = "sequence"
        else:
            data_dir = params["mpe_data_dir"]
            self.data_source = "mpe"
        if self.data_source == "snli":
            self.seq_len = 82 # max in SNLI train
        else:
            self.seq_len = 0
        if other_data is None:
            w_init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=12132015)
            self.curr_idx = 0
            self.word_to_idx = {}
            self.idx_to_word = {}
            self.embeddings_list = []
            self.load_vectors(vector_path=vector_path, vector_type=params["vector_src"])
            # oov vector
            self.embeddings_list.append(tf.get_variable(name="oov", shape=[self.embedding_dim], initializer=w_init, trainable=False))
            self.oov = "oov"
            self.word_to_idx[self.oov] = self.curr_idx
            self.idx_to_word[self.curr_idx] = self.oov
            self.curr_idx += 1
        else:
            self.embeddings_list = other_data.embeddings_list
            self.word_to_idx = other_data.word_to_idx
            self.idx_to_word = other_data.idx_to_word
            self.curr_idx = other_data.curr_idx
            self.oov = other_data.oov
        self.dataset = self.load_data(data_dir=data_dir)
        self.embeddings = tf.pack(self.embeddings_list)
        self.predictions = {}
        self.confidences = {}

    # tokenize punctuation (separate from words with whitespace)
    def clean_sequence_to_words(self, sequence):
        sequence = sequence.lower()
        for punctuation in self.punctuations:
            sequence = sequence.replace(punctuation, " {} ".format(punctuation))
        sequence = sequence.replace("  ", " ")
        sequence = sequence.replace("   ", " ")
        sequence = sequence.split(" ")
        todelete = ["", " ", "  "]
        for i, elt in enumerate(sequence):
            if elt in todelete:
                sequence.pop(i)
        return sequence

    def load_vectors(self, vector_path, vector_type):
        print "\nLoading vectors:"
        if vector_type == "glove" or vector_type == "w2v" or vector_type == "word2vec":
            for line in gzip.open(vector_path):
                tokens = line.strip().split()
                temp = []
                for t in tokens[1:]:
                    temp.append(float(t))
                self.word_to_idx[tokens[0]] = self.curr_idx
                self.idx_to_word[self.curr_idx] = tokens[0]
                self.embeddings_list.append(tf.Variable(temp, dtype=tf.float32, trainable=False))
                self.curr_idx += 1
        else:
            print "*******\n*******\n*******\nVECTORS NOT LOADED\n*******\n*******\n*******"
        print "loading: done"

    def load_data(self, data_dir):
        dataset_temp = {}
        print "\nLoading dataset:"
        for type_set_data in self.load_splits:
            type_set = self.split_map[type_set_data]
            if self.data_source == "snli":
                df = pd.read_csv(os.path.join(data_dir, "snli_1.0_{}.txt".format(type_set_data)), delimiter="\t")
                dataset_temp[type_set] = {"premises": df[["sentence1"]].values, "hypothesis": df[["sentence2"]].values,
                                          "labels": df[["gold_label"]].values}
            elif self.data_source == "mpe":
                if self.data_format == "multipremise":
                    dataset_temp[type_set] = {"ids": [], "premise1": [], "premise2": [], "premise3": [], "premise4": [],
                                              "hypothesis": [], "labels": [], "full": []}
                    with open(os.path.join(data_dir, "{}_header.txt".format(type_set_data))) as in_file:
                        for line in in_file:
                            if line.startswith("ID"):
                                continue
                            tokens = line.strip().split("\t")
                            dataset_temp[type_set]["ids"].append([tokens[0]])
                            dataset_temp[type_set]["premise1"].append([tokens[1].split("/")[1]])
                            dataset_temp[type_set]["premise2"].append([tokens[2].split("/")[1]])
                            dataset_temp[type_set]["premise3"].append([tokens[3].split("/")[1]])
                            dataset_temp[type_set]["premise4"].append([tokens[4].split("/")[1]])
                            dataset_temp[type_set]["hypothesis"].append([tokens[5]])
                            dataset_temp[type_set]["full"].append([tokens[1].split("/")[1] + " " + tokens[2].split("/")[1] + " " + tokens[3].split("/")[1] + " " + tokens[4].split("/")[1] + " " + tokens[5]])
                            dataset_temp[type_set]["labels"].append([tokens[9].split(":")[1]])
                elif self.data_format == "sequence":
                    dataset_temp[type_set] = {"ids": [], "premises": [], "hypothesis": [], "labels": []}
                    with open(os.path.join(data_dir, "{}_concat.txt".format(type_set_data))) as in_file:
                        for line in in_file:
                            if line.startswith("ID"):
                                continue
                            tokens = line.strip().split("\t")
                            dataset_temp[type_set]["ids"].append([tokens[0]])
                            dataset_temp[type_set]["premises"].append([tokens[1]])
                            dataset_temp[type_set]["hypothesis"].append([tokens[2]])
                            dataset_temp[type_set]["labels"].append([tokens[3]])
        dataset = self.preprocess(dataset=dataset_temp)
        print "dataset: done\n"
        return dataset

    def preprocess(self, dataset):
        map_targets = {"neutral": 0, "entailment": 1, "contradiction": 2}
        target_list = ["neutral", "entailment", "contradiction"]
        if self.data_source == "mpe" and self.data_format == "multipremise":
            map_targets = {"unk": 0, "yes": 1, "no": 2}
            target_list = ["unk", "yes", "no"]
        tokenized_dataset = dict((type_set, {"ids": [], "premise_sents": [], "premise_idx": [], "hypothesis_sents": [],
                                             "hypothesis_idx": [], "labels": [], "premise_lengths": [],
                                             "hyp_lengths": []}) for type_set in dataset)
        print "tokenization:"
        if self.stage == "test":
            seq_split = "test"
        else:
            seq_split = "train"
        for i in range(len(dataset[seq_split]["labels"])):
            try:
                max_premise_length = 0
                max_full_length = 0
                if self.data_format == "sequence":
                    premises_tokens = [word for word in self.clean_sequence_to_words(dataset[seq_split]["premises"][i][0])]
                    max_premise_length = len(premises_tokens)
                elif self.data_format == "multipremise":
                    full_tokens = [word for word in
                                       self.clean_sequence_to_words(
                                           dataset[seq_split]["full"][i][0])]
                    max_full_length = max(max_full_length, len(full_tokens))
                    for j in range(4):
                        premises_tokens = [word for word in
                                           self.clean_sequence_to_words(
                                               dataset[seq_split]["premise" + str(j + 1)][i][0])]
                        max_premise_length = max(max_premise_length, len(premises_tokens))
                hypothesis_tokens = [word for word in
                                         self.clean_sequence_to_words(dataset[seq_split]["hypothesis"][i][0])]
                self.seq_len = max(self.seq_len, max(max_premise_length, len(hypothesis_tokens)))
                if self.data_format == "multipremise":
                    self.seq_len = max(self.seq_len, max_full_length)
            except:
                pass
        count = 0 # create SNLI ids
        for type_set in dataset:
            print "type_set:", type_set
            num_ids = len(dataset[type_set]["labels"])
            print "num_ids", num_ids
            for i in range(num_ids):
                try:
                    target = map_targets[dataset[type_set]["labels"][i][0]]
                    hypothesis_tokens = [word for word in
                                         self.clean_sequence_to_words(dataset[type_set]["hypothesis"][i][0])]
                    if self.data_format == "sequence":
                        premises_tokens = [word for word in self.clean_sequence_to_words(dataset[type_set]["premises"][i][0])]
                        tokenized_dataset[type_set]["premise_sents"].append(premises_tokens)
                        tokenized_dataset[type_set]["premise_idx"].append(self.sent_to_idx(premises_tokens))
                        tokenized_dataset[type_set]["premise_lengths"].append(len(premises_tokens))
                        if self.data_source == "snli":
                            tokenized_dataset[type_set]["ids"].append(count)
                        else:
                            tokenized_dataset[type_set]["ids"].append(dataset[type_set]["ids"][i][0])
                    elif self.data_format == "multipremise":
                        premise_sents_temp = []
                        premise_idx_temp = []
                        premise_lengths_temp = []
                        for j in range(4):
                            premises_tokens = [word for word in
                                               self.clean_sequence_to_words(
                                                   dataset[type_set]["premise" + str(j + 1)][i][0])]
                            premise_sents_temp.append(premises_tokens)
                            sent_idx = self.sent_to_idx(premises_tokens)
                            premise_idx_temp.append(sent_idx)
                            premise_lengths_temp.append(len(premises_tokens))
                        tokenized_dataset[type_set]["premise_sents"].append(premise_sents_temp)
                        tokenized_dataset[type_set]["premise_idx"].append(premise_idx_temp)
                        tokenized_dataset[type_set]["premise_lengths"].append(premise_lengths_temp)
                        tokenized_dataset[type_set]["ids"].append(dataset[type_set]["ids"][i][0])
                    tokenized_dataset[type_set]["hypothesis_sents"].append(hypothesis_tokens)
                    hyp_idx = self.sent_to_idx(hypothesis_tokens)
                    tokenized_dataset[type_set]["hypothesis_idx"].append(hyp_idx)
                    tokenized_dataset[type_set]["hyp_lengths"].append(len(hypothesis_tokens))
                    l_temp = np.zeros(shape=[len(target_list)], dtype=np.float32)
                    l_temp[target] = 1
                    tokenized_dataset[type_set]["labels"].append(l_temp)
                    count += 1
                except:
                    pass
                sys.stdout.write("\rid: {}/{}      ".format(i + 1, num_ids))
                sys.stdout.flush()
            print ""
        print "tokenization: done"
        return tokenized_dataset

    def sent_to_idx(self, words):
        p_seq = copy.deepcopy(words)
        preprocessed = []
        diff_size = len(p_seq) - self.seq_len
        if diff_size > 0:
            start_index = 0
            p_seq = p_seq[start_index: (start_index + self.seq_len)]
        for word in p_seq:
            preprocessed.append(self.lookup_word_idx(word))
        for i in range(self.seq_len - len(preprocessed)):
            preprocessed.append(0)
        return preprocessed

    def lookup_word_idx(self, word):
        try:
            idx = self.word_to_idx[word]
        except KeyError:
            idx = self.word_to_idx[self.oov]
        return idx

    def get_size(self, type_set):
        return len(self.dataset[type_set]["labels"])

    def init_eval(self, type_set):
        self.predictions[type_set] = {}
        self.confidences[type_set] = {}
        for i in self.dataset[type_set]["ids"]:
            self.predictions[type_set][i] = []
            self.confidences[type_set][i] = []

    def update_eval(self, type_set, ids, predictions, confidences):
        for idx, id in enumerate(ids):
            self.predictions[type_set][id].append(predictions[idx])
            self.confidences[type_set][id].append(confidences[idx])

    def get_top_10_conf(self, conf, type_set):
        label_conf_10 = {}
        for label in {"entailment": 0, "neutral": 0, "contradiction": 0}:
            conf_sorted = sorted(conf[label], key=conf[label].get, reverse=True)
            conf_10 = []
            sum = {0:0, 1:0, 2:0}
            max = min(10, len(conf_sorted))
            if max > 0:
                for i in range(max):
                    eval_id = conf_sorted[i]
                    confidences = np.ndarray.tolist(self.confidences[type_set][eval_id][0])
                    conf_10.append((eval_id, confidences))
                    for idx, val in enumerate(confidences):
                        sum[idx] += val
                for key in sum:
                    sum[key] /= float(max)
            conf_10.append((sum))
            label_conf_10[label] = conf_10
        return label_conf_10

    def summarize_eval(self, type_set):
        correct = 0
        total_pred = 0
        label_map = {0: "neutral", 1: "entailment", 2: "contradiction"}
        true_pos = {"entailment": 0, "neutral": 0, "contradiction": 0}
        false_pos = {"entailment": 0, "neutral": 0, "contradiction": 0}
        false_neg = {"entailment": 0, "neutral": 0, "contradiction": 0}
        incorrect_conf = {"entailment":{}, "neutral":{}, "contradiction":{}}
        correct_conf = {"entailment":{}, "neutral":{}, "contradiction":{}}
        correct_label = {"entailment": 0, "neutral": 0, "contradiction": 0}
        total_label = {"entailment": 0, "neutral": 0, "contradiction": 0}
        correct_ids = []
        for idx, id in enumerate(self.dataset[type_set]["ids"]):
            if len(self.predictions[type_set][id]) == 0:
                continue
            total_pred += 1
            pred_label_id = self.predictions[type_set][id][0]
            true_label_id = np.where(self.dataset[type_set]["labels"][idx] == 1)[0][0]
            total_label[label_map[true_label_id]] += 1
            if pred_label_id == true_label_id:
                correct += 1
                correct_label[label_map[true_label_id]] += 1
                true_pos[label_map[true_label_id]] += 1
                correct_conf[label_map[true_label_id]][id] = self.confidences[type_set][id][0][true_label_id]
                correct_ids.append(id)
            else:
                false_pos[label_map[pred_label_id]] += 1
                false_neg[label_map[true_label_id]] += 1
                incorrect_conf[label_map[true_label_id]][id] = self.confidences[type_set][id][0][pred_label_id]
        precision = {"entailment": 0, "neutral": 0, "contradiction": 0}
        recall = {"entailment": 0, "neutral": 0, "contradiction": 0}
        for label in ["entailment", "neutral", "contradiction"]:
            prec_sum = 1.0 * (true_pos[label] + false_pos[label])
            if prec_sum > 0:
                precision[label] = true_pos[label] / prec_sum
            else:
                precision[label] = 0.0
            recall_sum = 1.0 * (true_pos[label] + false_neg[label])
            if recall_sum > 0:
                recall[label] = true_pos[label] / recall_sum
            else:
                recall[label] = 0.0
        f1 = {}
        for label in precision:
            if (precision[label] + recall[label]) > 0:
                f1[label] = (2 * precision[label] * recall[label]) / (precision[label] + recall[label])
            else:
                f1[label] = 0.0
        acc_label = {}
        for label in correct_label:
            acc_label[label] = correct_label[label] / float(total_label[label])
        return precision, recall, f1, 100 * (correct * 1.0 / total_pred), correct_ids, self.get_top_10_conf(correct_conf, type_set), self.get_top_10_conf(incorrect_conf, type_set), total_pred, acc_label

    def get_batch(self, data_split, indices):
        if self.data_format == "sequence": # assume same number of premises and hypotheses
            batch = {
                "premise_sents": [self.dataset[data_split]["premise_sents"][i] for i in indices],
                "premise_lengths": [self.dataset[data_split]["premise_lengths"][i] for i in indices],
                "premises": [self.dataset[data_split]["premise_idx"][i] for i in indices],
                "hyp_lengths": [self.dataset[data_split]["hyp_lengths"][i] for i in indices],
                "hypothesis_sents": [self.dataset[data_split]["hypothesis_sents"][i] for i in indices],
                "hypotheses": [self.dataset[data_split]["hypothesis_idx"][i] for i in indices],
                "labels": [self.dataset[data_split]["labels"][i] for i in indices],
                "ids": [self.dataset[data_split]["ids"][i] for i in indices]
            }
            return batch
        elif self.data_format == "multipremise": # premises is 4x longer list compared to hypothesis list
            batch = {
                "premise1_sents": [self.dataset[data_split]["premise_sents"][i][0] for i in indices],
                "premise1_lengths": [self.dataset[data_split]["premise_lengths"][i][0] for i in indices],
                "premises1": [self.dataset[data_split]["premise_idx"][i][0] for i in indices],
                "premise2_sents": [self.dataset[data_split]["premise_sents"][i][1] for i in indices],
                "premise2_lengths": [self.dataset[data_split]["premise_lengths"][i][1] for i in indices],
                "premises2": [self.dataset[data_split]["premise_idx"][i][1] for i in indices],
                "premise3_sents": [self.dataset[data_split]["premise_sents"][i][2] for i in indices],
                "premise3_lengths": [self.dataset[data_split]["premise_lengths"][i][2] for i in indices],
                "premises3": [self.dataset[data_split]["premise_idx"][i][2] for i in indices],
                "premise4_sents": [self.dataset[data_split]["premise_sents"][i][3] for i in indices],
                "premise4_lengths": [self.dataset[data_split]["premise_lengths"][i][3] for i in indices],
                "premises4": [self.dataset[data_split]["premise_idx"][i][3] for i in indices],
                "hyp_lengths": [self.dataset[data_split]["hyp_lengths"][i] for i in indices],
                "hypothesis_sents": [self.dataset[data_split]["hypothesis_sents"][i] for i in indices],
                "hypotheses": [self.dataset[data_split]["hypothesis_idx"][i] for i in indices],
                "labels": [self.dataset[data_split]["labels"][i] for i in indices],
                "ids": [self.dataset[data_split]["ids"][i] for i in indices]
            }
            return batch