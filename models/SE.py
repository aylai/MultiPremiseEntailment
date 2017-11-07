import tensorflow as tf
from util.Layer import *
import os
import time
import progressbar
import datetime
import random
import numpy as np
import copy
from util.results import *

tf.set_random_seed(20170302)
np.random.seed(20170302)
random.seed(20170302)


class Model(object):

    def __init__(self, data_mpe, parameters, data_snli=None, load_model=None, premise_strategy="concat", num_hidden=1):
        self.data_mpe = data_mpe
        self.data_snli = data_snli
        self.parameters = parameters
        self.num_hidden = num_hidden
        self.premise_strategy = premise_strategy

        """ Model Definition """
        self.dropout_ph = tf.placeholder(tf.float32, name="dropout")
        self.labels_ph = tf.placeholder(tf.int32, shape=[None, parameters["num_classes"]], name="labels")
        self.premise1_lengths = tf.placeholder(tf.int32, shape=[parameters["batch_size"]], name="premise1_lengths")
        self.premise2_lengths = tf.placeholder(tf.int32, shape=[parameters["batch_size"]], name="premise2_lengths")
        self.premise3_lengths = tf.placeholder(tf.int32, shape=[parameters["batch_size"]], name="premise3_lengths")
        self.premise4_lengths = tf.placeholder(tf.int32, shape=[parameters["batch_size"]], name="premise4_lengths")
        self.premises1_ph = tf.placeholder(tf.int32,
                                           shape=[parameters["batch_size"], None],
                                           name="premises1")
        self.premises2_ph = tf.placeholder(tf.int32,
                                           shape=[parameters["batch_size"], None],
                                           name="premises2")
        self.premises3_ph = tf.placeholder(tf.int32,
                                           shape=[parameters["batch_size"], None],
                                           name="premises3")
        self.premises4_ph = tf.placeholder(tf.int32,
                                           shape=[parameters["batch_size"], None],
                                           name="premises4")
        self.hyp_lengths = tf.placeholder(tf.int32, shape=[parameters["batch_size"]], name="hypothesis_lengths")
        self.hypotheses_ph = tf.placeholder(tf.int32,
                                            shape=[parameters["batch_size"], None],
                                            name="hypothesis")

        sentence_pair_mpe, sentence_pair_snli = self.sentence_pair_rep()

        optimizer = tf.train.AdamOptimizer(self.parameters["learning_rate"])

        """ MPE MODEL """
        self.confidence_mpe = tf.nn.softmax(sentence_pair_mpe)
        self.loss_mpe = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(sentence_pair_mpe, self.labels_ph))
        self.predictions_mpe = tf.argmax(sentence_pair_mpe, 1)
        self.correct_mpe = tf.equal(tf.argmax(sentence_pair_mpe, 1), tf.argmax(self.labels_ph, 1))
        self.train_op_mpe = optimizer.minimize(self.loss_mpe)

        """ SNLI MODEL """
        self.confidence_snli = tf.nn.softmax(sentence_pair_snli)
        self.loss_snli = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(sentence_pair_snli, self.labels_ph))
        self.predictions_snli = tf.argmax(sentence_pair_snli, 1)
        self.correct_snli = tf.equal(tf.argmax(sentence_pair_snli, 1), tf.argmax(self.labels_ph, 1))
        self.train_op_snli = optimizer.minimize(self.loss_snli)

        modeldir = os.path.join(parameters["run_dir"], parameters["exp_name"])
        self.logdir = os.path.join(modeldir, "log")
        self.savepath = os.path.join(modeldir, "save")
        if self.data_mpe.data_source == "mpe" and self.parameters["stage"] == "test":
            self.pred_file = open(os.path.join(modeldir, "predictions_" + self.parameters["test_split"] + ".txt"), "w")
        self.train_file = open(os.path.join(modeldir, "training.txt"), "w")

        self.saver = tf.train.Saver(max_to_keep=50)
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        if load_model is not None:
            loadpath = os.path.join(modeldir, load_model)
            print "Restoring ", loadpath
            self.saver.restore(self.sess, loadpath)
        else:
            print "Initializing"
            self.sess.run(init_op)

    def sentence_pair_rep(self):

        with tf.variable_scope("lstm"):
            premise1_emb = tf.nn.embedding_lookup(self.data_mpe.embeddings, self.premises1_ph)
            premise2_emb = tf.nn.embedding_lookup(self.data_mpe.embeddings, self.premises2_ph)
            premise3_emb = tf.nn.embedding_lookup(self.data_mpe.embeddings, self.premises3_ph)
            premise4_emb = tf.nn.embedding_lookup(self.data_mpe.embeddings, self.premises4_ph)
            hypothesis_emb = tf.nn.embedding_lookup(self.data_mpe.embeddings, self.hypotheses_ph)

            with tf.variable_scope("lstm_p"):
                lstm_p = tf.nn.rnn_cell.LSTMCell(self.parameters["lstm_dim"], state_is_tuple=True)
                lstm_p = tf.nn.rnn_cell.DropoutWrapper(lstm_p, output_keep_prob=self.dropout_ph)
                lstm_p = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_p] * self.parameters["multicell"], state_is_tuple=True)

                lstm_h = tf.nn.rnn_cell.LSTMCell(self.parameters["lstm_dim"], state_is_tuple=True)
                lstm_h = tf.nn.rnn_cell.DropoutWrapper(lstm_h, output_keep_prob=self.dropout_ph)
                lstm_h = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_h] * self.parameters["multicell"], state_is_tuple=True)

                outputs1_p, fstate1_p = tf.nn.dynamic_rnn(lstm_p, premise1_emb, sequence_length=self.premise1_lengths,
                                                        dtype=tf.float32)
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    outputs1_h, fstate1_h = tf.nn.dynamic_rnn(lstm_h, hypothesis_emb, initial_state=fstate1_p,
                                                              sequence_length=self.hyp_lengths, dtype=tf.float32)
                    outputs2_p, fstate2_p = tf.nn.dynamic_rnn(lstm_p, premise2_emb, sequence_length=self.premise2_lengths,
                                                          dtype=tf.float32)
                    outputs2_h, fstate2_h = tf.nn.dynamic_rnn(lstm_h, hypothesis_emb, initial_state=fstate2_p,
                                                              sequence_length=self.hyp_lengths, dtype=tf.float32)
                    outputs3_p, fstate3_p = tf.nn.dynamic_rnn(lstm_p, premise3_emb, sequence_length=self.premise3_lengths,
                                                          dtype=tf.float32)
                    outputs3_h, fstate3_h = tf.nn.dynamic_rnn(lstm_h, hypothesis_emb, initial_state=fstate3_p,
                                                              sequence_length=self.hyp_lengths, dtype=tf.float32)
                    outputs4_p, fstate4_p = tf.nn.dynamic_rnn(lstm_p, premise4_emb, sequence_length=self.premise4_lengths,
                                                          dtype=tf.float32)
                    outputs4_h, fstate4_h = tf.nn.dynamic_rnn(lstm_h, hypothesis_emb, initial_state=fstate4_p,
                                                              sequence_length=self.hyp_lengths, dtype=tf.float32)

            # if self.parameters["num_hidden"] == 0:
            output_layer1 = ff_w((2 * self.parameters["multicell"] * self.parameters["lstm_dim"]),
                                 self.parameters["num_classes"], name="Output1")

            output_bias1 = ff_b(self.parameters["num_classes"], "OutputBias1")
            logits1 = tf.matmul(tf.concat(1, [tf.concat(1, [f.h for f in fstate1_p]), tf.concat(1, [f.h for f in fstate1_h])]), output_layer1) + output_bias1
            logits2 = tf.matmul(tf.concat(1, [tf.concat(1, [f.h for f in fstate2_p]), tf.concat(1, [f.h for f in fstate2_h])]), output_layer1) + output_bias1
            logits3 = tf.matmul(tf.concat(1, [tf.concat(1, [f.h for f in fstate3_p]), tf.concat(1, [f.h for f in fstate3_h])]), output_layer1) + output_bias1
            logits4 = tf.matmul(tf.concat(1, [tf.concat(1, [f.h for f in fstate4_p]), tf.concat(1, [f.h for f in fstate4_h])]), output_layer1) + output_bias1
            logits = tf.add_n([logits1, logits2, logits3, logits4])
            return logits, logits1

    def train(self, train, dev):
        print("train data size: %d" % len(self.data_mpe.dataset[train]["labels"]))
        if self.data_snli is not None:
            print("train data size: %d" % (len(self.data_snli.dataset[train]["labels"])))
        best_dev_accuracy = 0.0
        total_loss = 0.0
        timestamp = time.time()
        epoch_data = []
        if self.parameters["snli_train"] is None and self.parameters["snli_pretrain"] is None:
            for i in range(self.parameters["num_epochs"]):
                epoch_data.append("mpe")
        else:
            if self.parameters["snli_pretrain"]:
                for i in range(self.parameters["num_pretraining_epochs"]):
                    epoch_data.append("snli")
                for i in range(self.parameters["num_epochs"]):
                    epoch_data.append("mpe")
            else: # train SNLI only
                for i in range(self.parameters["num_epochs"]):
                    epoch_data.append("snli")
        for epoch, data_source in enumerate(epoch_data):
            print("data: " + data_source)
            self.train_file.write("data: " + data_source + "\n")
            if data_source == "snli":
                indices = list(range(self.data_snli.get_size(train)))
            else:
                indices = list(range(self.data_mpe.get_size(train)))
            random.shuffle(indices)
            steps = len(indices) / self.parameters["batch_size"]
            bar = progressbar.ProgressBar(maxval=steps / 10 + 1,
                                          widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()])
            bar.start()
            for step in range(steps):
                r_ind = indices[(step * self.parameters["batch_size"]):((step + 1) * self.parameters["batch_size"])]
                if data_source == "snli":
                    """ SNLI step """
                    train_batch = self.data_snli.get_batch(train, r_ind, self.parameters["batch_size"])
                    feed_dict = {self.premises1_ph: train_batch["premises"],
                                 self.hypotheses_ph: train_batch["hypotheses"],
                                 self.labels_ph: train_batch["labels"],
                                 self.premise1_lengths: train_batch["premise_lengths"],
                                 self.hyp_lengths: train_batch["hyp_lengths"],
                                 self.dropout_ph: self.parameters["dropout"]}
                    _, train_loss, train_pred, conf = self.sess.run(
                        [self.train_op_snli, self.loss_snli, self.predictions_snli, self.confidence_snli],
                        feed_dict=feed_dict)
                else:
                    """ MPE step """
                    train_batch = self.data_mpe.get_batch(train, r_ind)
                    feed_dict = {self.premises1_ph: train_batch["premises1"],
                                 self.premises2_ph: train_batch["premises2"],
                                 self.premises3_ph: train_batch["premises3"],
                                 self.premises4_ph: train_batch["premises4"],
                                 self.hypotheses_ph: train_batch["hypotheses"], self.labels_ph: train_batch["labels"],
                                 self.premise1_lengths: train_batch["premise1_lengths"],
                                 self.premise2_lengths: train_batch["premise2_lengths"],
                                 self.premise3_lengths: train_batch["premise3_lengths"],
                                 self.premise4_lengths: train_batch["premise4_lengths"],
                                 self.hyp_lengths: train_batch["hyp_lengths"],
                                 self.dropout_ph: self.parameters["dropout"]}
                    _, train_loss, train_pred, conf = self.sess.run(
                        [self.train_op_mpe, self.loss_mpe, self.predictions_mpe, self.confidence_mpe],
                        feed_dict=feed_dict)
                total_loss += train_loss
                if step % 100 == 0:
                    bar.update(step / 10 + 1)
            bar.finish()
            dev_loss, dev_precision, dev_recall, dev_f1, dev_accuracy, dev_correct_ids, dev_corr_10, dev_incorrect_10, total_items, acc_label = self.eval(
                self.data_mpe, dev, True)  # eval on full dev for printing (but not tensorboard)
            train_loss, train_precision, train_recall, train_f1, train_accuracy, train_correct_ids, corr_10, incorrect_10, total_items, acc_label = self.eval(
                self.data_mpe, train, False)
            current_time = time.time()
            iter_str = (
                "Iter %3d  Train Loss %-8.3f  Dev Loss %-8.3f  Sample Train Acc %-6.2f  Dev Acc %-6.2f  Time %-5.2f at %s" %
                (epoch, total_loss, dev_loss, train_accuracy, dev_accuracy,
                 (current_time - timestamp) / 60.0, str(datetime.datetime.now())))
            print(iter_str)
            self.train_file.write(iter_str + "\n")
            out_str, file_str = format_epoch_string(train_precision, train_recall, train_f1, dev_precision, dev_recall,
                                                    dev_f1)
            print(out_str)
            self.train_file.write(file_str)
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                self.saver.save(self.sess, save_path=self.savepath + "_best", global_step=epoch)
            self.saver.save(self.sess, save_path=self.savepath, global_step=epoch)
            total_loss = 0.0
        self.train_file.close()

    def eval(self, data_source, eval_data, full):
        data_source.init_eval(eval_data)
        loss = []
        correct = []
        labels = []
        indices = range(data_source.get_size(eval_data))
        if not full:
            random.shuffle(indices)
        count = 0
        steps = len(indices) / self.parameters["batch_size"]
        if len(indices) % self.parameters["batch_size"] != 0:
            steps += 1
        for i in range(steps):
            if count > 0 and not full:
                break
            count += 1
            padded = 0
            while self.parameters["batch_size"]*(i+1) > len(indices):
                indices.append(indices[-1])
                padded += 1
            ind = indices[self.parameters["batch_size"] * i:self.parameters["batch_size"] * (i + 1)]
            eval_batch = data_source.get_batch(eval_data, ind)
            labels.extend(eval_batch["labels"])
            l, p, corr, conf = self.sess.run([self.loss_mpe, self.predictions_mpe, self.correct_mpe,
                                                       self.confidence_mpe],
                                          feed_dict={self.premises1_ph: eval_batch["premises1"],
                                                     self.premises2_ph: eval_batch["premises2"],
                                                     self.premises3_ph: eval_batch["premises3"],
                                                     self.premises4_ph: eval_batch["premises4"],
                                                     self.hypotheses_ph: eval_batch["hypotheses"],
                                                     self.labels_ph: eval_batch["labels"],
                                                     self.premise1_lengths: eval_batch["premise1_lengths"],
                                                     self.premise2_lengths: eval_batch["premise2_lengths"],
                                                     self.premise3_lengths: eval_batch["premise3_lengths"],
                                                     self.premise4_lengths: eval_batch["premise4_lengths"],
                                                     self.hyp_lengths: eval_batch["hyp_lengths"],
                                                     self.dropout_ph: 1.0})
            loss.append(l)
            ids = copy.deepcopy(eval_batch["ids"])
            if padded > 0:
                p = p[:-1 * padded]
                corr = corr[:-1 * padded]
                ids = ids[:-1 * padded]
                conf = conf[:-1 * padded]
            correct.extend(corr)
            data_source.update_eval(eval_data, ids, p, conf)
        if self.parameters["stage"] == "test":
            for idx, id in enumerate(data_source.dataset[self.parameters["test_split"]]["ids"]):
                self.pred_file.write(id + " ")
                if len(data_source.confidences[self.parameters["test_split"]][id]) > 0:
                    for pair_pred in data_source.confidences[self.parameters["test_split"]][id]:
                        for val in pair_pred:
                            self.pred_file.write(str(val) + " ")
                self.pred_file.write("\n")
        precision, recall, f1, acc, correct_ids, corr_10, incorrect_10, total_items, acc_label = data_source.summarize_eval(eval_data)
        return np.sum(loss), precision, recall, f1, acc, correct_ids, corr_10, incorrect_10, total_items, acc_label

    def test(self, split):
        test_loss, test_precision, test_recall, test_f1, test_accuracy, test_correct_ids, test_corr_10, test_incorrect_10, num_items, acc_label = self.eval(
            self.data_mpe, split, True)  # eval on full dev for printing (but not tensorboard)
        print(format_eval_string(test_loss, split, test_accuracy, acc_label, test_precision, test_recall, test_f1,
                                 test_correct_ids))