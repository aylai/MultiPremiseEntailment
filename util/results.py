

def format_epoch_string(train_precision, train_recall, train_f1, dev_precision, dev_recall, dev_f1):
    out_str = ""
    file_str = ""
    out_str += "train precision:\n"
    file_str += "train precision:\n"
    results = ""
    for label in ["entailment", "neutral", "contradiction"]:
        results += "\t%s: %-6.2f" % (label, train_precision[label])
    out_str += results + "\n"
    file_str += results + "\n"
    out_str += "train recall:\n"
    file_str += "train recall:\n"
    results = ""
    for label in ["entailment", "neutral", "contradiction"]:
        results += "\t%s: %-6.2f" % (label, train_recall[label])
    out_str += results + "\n"
    file_str += results + "\n"
    out_str += "train F1:\n"
    file_str += "train F1:\n"
    results = ""
    for label in ["entailment", "neutral", "contradiction"]:
        results += "\t%s: %-6.2f" % (label, train_f1[label])
    out_str += results + "\n"
    file_str += results + "\n"
    out_str += "dev precision:\n"
    file_str += "dev precision:\n"
    results = ""
    for label in ["entailment", "neutral", "contradiction"]:
        results += "\t%s: %-6.2f" % (label, dev_precision[label])
    out_str += results + "\n"
    file_str += results + "\n"
    out_str += "dev recall:\n"
    file_str += "dev recall:\n"
    results = ""
    for label in ["entailment", "neutral", "contradiction"]:
        results += "\t%s: %-6.2f" % (label, dev_recall[label])
    out_str += results + "\n"
    file_str += results + "\n"
    out_str += "dev F1:\n"
    file_str += "dev F1:\n"
    results = ""
    for label in ["entailment", "neutral", "contradiction"]:
        results += "\t%s: %-6.2f" % (label, dev_f1[label])
    out_str += results + "\n"
    file_str += results + "\n"
    return out_str, file_str


def format_eval_string(loss, split, accuracy, acc_label, precision, recall, f1, correct_ids):
    out_str = ""
    out_str += ("Loss %-8.3f  %s Acc %-6.2f" % (loss, split, accuracy))
    out_str += "test accuracy:\n"
    results = ""
    for label in ["entailment", "neutral", "contradiction"]:
        results += "\t%s: %-6.2f" % (label, acc_label[label])
    out_str += results + "\n"
    out_str += "test precision" + precision + "\n"
    out_str += "test recall" + recall + "\n"
    out_str += "test precision:\n"
    results = ""
    for label in ["entailment", "neutral", "contradiction"]:
        results += "\t%s: %-6.2f" % (label, precision[label])
    out_str += results + "\n"
    out_str += "test recall:\n"
    results = ""
    for label in ["entailment", "neutral", "contradiction"]:
        results += "\t%s: %-6.2f" % (label, recall[label])
    out_str += results + "\n"
    out_str += "test F1:\n"
    results = ""
    for label in ["entailment", "neutral", "contradiction"]:
        results += "\t%s: %-6.2f" % (label, f1[label])
    out_str += results + "\n"
    out_str += "**\n"
    for test_id in correct_ids:
        out_str += test_id + " "
    out_str += "**"
    return out_str