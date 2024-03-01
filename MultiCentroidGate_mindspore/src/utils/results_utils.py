from copy import deepcopy
from .metrics import accuracy
import json
import jsbeautifier
import mindspore


def generate_report(ypred, ytrue, increments):
    all_acc = {"top1": {"per_task": []}, "top5": {"per_task": []}}
    top1, top5 = accuracy(ypred, ytrue, (1, 5))
    all_acc["top1"]["total"] = round(top1, 3)
    all_acc["top5"]["total"] = round(top5, 3)
    start, end = 0, 0
    for i in range(len(increments)):
        start = end
        end += increments[i]
        idxes = mindspore.ops.where(mindspore.ops.logical_and(ytrue >= start, ytrue < end), 
                                    mindspore.numpy.arange(ytrue.shape[0]),
                                    mindspore.numpy.full(ytrue.shape, -1))
        idxes = idxes[idxes != -1]
        top1, top5 = accuracy(ypred[idxes], ytrue[idxes], (1, 5))  
        all_acc["top1"]["per_task"].append(round(top1, 3))
        all_acc["top5"]["per_task"].append(round(top5, 3))
    return all_acc


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

def del_unjsonable(d):
    import json
    dcopy = deepcopy(d)
    for k, v in d.items():
        if not is_jsonable(v):
            del dcopy[k]
    return dcopy

def to_json(j):
    options = jsbeautifier.default_options()
    options.indent_size = 2
    return jsbeautifier.beautify(json.dumps(j), options)

def compute_avg_inc_acc(results):
    top1_tasks_accuracy = [r['top1']["total"] for r in results]
    top1acc = sum(top1_tasks_accuracy) / len(top1_tasks_accuracy)
    if "top5" in results[0].keys():
        top5_tasks_accuracy = [r['top5']["total"] for r in results]
        top5acc = sum(top5_tasks_accuracy) / len(top5_tasks_accuracy)
    else:
        top5acc = None
    return top1acc, top5acc