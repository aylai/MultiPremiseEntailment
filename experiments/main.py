import sys, os
sys.path.insert(0,os.getcwd())
from models import LSTM
from models import SE
from util.load_data import *
import argparse
import time

dirname, filename = os.path.split(os.path.abspath(__file__))
GIT_DIR = "/".join(dirname.split("/")[:-1])

DATA_DIR = os.path.join(GIT_DIR, "data")
RUNS_DIR = os.path.join(GIT_DIR, "runs")

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="active this flag to train the model")
parser.add_argument("--test", action="store_true", help="active this flag to test the model")
parser.add_argument("--train_split", type=str, default="train", help="data split to train model")
parser.add_argument("--dev_split", type=str, default="dev", help="data split for dev evaluation")
parser.add_argument("--test_split", type=str, default="test", help="data split to evaluate")
parser.add_argument("--saved_model", help="name of saved model")
parser.add_argument("--mpe_data_dir", default="mpe")
parser.add_argument("--snli_data_dir", default="snli")
parser.add_argument("--vector_type", default="glove", help="specify vector type glove/word2vec/none")
parser.add_argument("--word2vec_path", type=str, default="/home/aylai2/Projects/MPEModel/data/snli/word2vec.snli.txt.gz",
                    help="path to the pretrained Word2Vect .bin file")
parser.add_argument("--glove_path", type=str, default="/home/aylai2/Projects/MPEModelExperiments/data/mpe/glove.ALL.txt.gz")
parser.add_argument("--model_name", type=str)
parser.add_argument("--model_type", type=str, default="lstm")
parser.add_argument("--data_format", type=str, default="multipremise", help="multipremise or sequence")
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--dropout", type=float, default=0.8)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--batch_size_mpe", type=int, default=4)
parser.add_argument("--batch_size_snli", type=int, default=280)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--num_classes", type=int, default=3, help="number of entailment classes")
parser.add_argument("--embedding_dim", type=int, default=300, help="vector dimension")
parser.add_argument("--lstm_dim", type=int, default=100, help="LSTM output dimension (k in the original paper)")
parser.add_argument("--multicell", type=int, default=1, help="number of multicell LSTM layers")
parser.add_argument("--snli_train", action="store_true")
parser.add_argument("--snli_pretrain", action="store_true")
parser.add_argument("--num_pretraining_epochs", type=int, default=0)

parser.add_argument("--num_hidden", type=int, default=1, help="number of hidden layers after LSTM")
parser.add_argument("--num_features", type=int, default=4)
args = parser.parse_args()

if args.model_name is None:
    print "Specify name of experiment"
    sys.exit(0)

params = {
    "top_dir": GIT_DIR,
    "run_dir": RUNS_DIR,
    "exp_name": args.model_name,
    "load_model_file": args.saved_model,
    "mpe_data_dir": os.path.join(DATA_DIR, args.mpe_data_dir),
    "snli_data_dir": os.path.join(DATA_DIR, args.snli_data_dir),

    "train_split": args.train_split,
    "dev_split": args.dev_split,
    "test_split": args.test_split,

    "model_type": args.model_type,
    "glove_file": args.glove_path,
    "w2v_file": args.word2vec_path,
    "vector_src": args.vector_type,
    "data_format": args.data_format,

    "embedding_dim": args.embedding_dim, # word embedding dim

    "batch_size": args.batch_size,
    "batch_size_mpe": args.batch_size_mpe,
    "batch_size_snli": args.batch_size_snli,
    "lstm_dim": args.lstm_dim,
    "dropout": args.dropout, # 1 = no dropout, 0.5 = dropout
    "multicell": args.multicell,
    "snli_train": args.snli_train,
    "snli_pretrain": args.snli_pretrain,
    "learning_rate": args.learning_rate,
    "num_epochs": args.num_epochs,
    "num_pretraining_epochs": args.num_pretraining_epochs,
    "num_classes": args.num_classes,
    "num_hidden": args.num_hidden,
    "num_features": args.num_features,
}

if args.train:
    params["stage"] = "train"
elif args.test:
    params["stage"] = "test"
else:
    print "Choose to train or test model."
    sys.exit(0)

if not os.path.exists(params["run_dir"]):
    os.mkdir(params["run_dir"])
modeldir = os.path.join(params["run_dir"], params["exp_name"])
if not os.path.exists(modeldir):
    os.mkdir(modeldir)
logdir = os.path.join(modeldir, "log")
if not os.path.exists(logdir):
    os.mkdir(logdir)

# save parameters
with open(os.path.join(modeldir, params["exp_name"] + ".params"), "w") as param_file:
    for key, parameter in params.iteritems():
        param_file.write("{}: {}".format(key, parameter) + "\n")
        print(key, parameter)

start = time.time()

if params["vector_src"] == "glove":
    vector_path = params["glove_file"]
elif params["vector_src"] == "w2v":
    vector_path = params["w2v_file"]
dataset = Data(params) # loads MPE data
snli_dataset = None
if params["snli_train"] is not None or params["snli_pretrain"] is not None:
    snli_dataset = Data(params, other_data=dataset) # loads SNLI data

if params["stage"] == "train":
    model = None
    if params["model_type"] == "lstm":
        model = LSTM.Model(dataset, params, data_snli=snli_dataset)
    elif params["model_type"] == "se":
        model = SE.Model(dataset, params, data_snli=snli_dataset)
    if model is not None:
        model.train("train", "dev")
elif params["stage"] == "test":
    model = None
    if params["model_type"] == "lstm":
        model = LSTM.Model(dataset, params, load_model=params["load_model_file"])
    elif params["model_type"] == "se":
        model = SE.Model(dataset, params, load_model=params["load_model_file"])
    if model is not None and params["model_type"] != "joint":
        model.test("test")


end = time.time() - start
m, s = divmod(end, 60)
h, m = divmod(m, 60)
print "%d:%02d:%02d" % (h, m, s)
