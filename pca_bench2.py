import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import faiss
import knowhere
import json
import time
import random


def read_bin(file_path, dtype):
    data = np.fromfile(file_path).view('int32')
    n, d = data[:2]
    return data[2:].view(dtype).reshape(n, d)


def read_vecs(file_path, dtype):
    data = np.fromfile(file_path).view('int32')
    d = data[0]
    return data.reshape(-1, d+1)[:, 1:].view(dtype)


def read_data(file_path, dtype, format):
    if format == 'bin':
        return read_bin(file_path, dtype)
    elif format == 'vecs':
        return read_vecs(file_path, dtype)


datasetname = 'cohere'
dir = '../dataset'
format = "bin"
metric = "L2"
normalize = True
topk = 100

xb = read_data(f'{dir}/{datasetname}/{datasetname}_base.f{format}',
               'float32', format)
xq = read_data(
    f'{dir}/{datasetname}/{datasetname}_query.f{format}', 'float32', format)
gt = read_data(
    f'{dir}/{datasetname}/{datasetname}_groundtruth.i{format}', 'int32', format)[:, :topk]
if normalize:
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    xq = xq / np.linalg.norm(xq, axis=1, keepdims=True)
n, d = xb.shape
nq = xq.shape[0]
print(f'n = {n}, d = {d}, nq = {nq}')


version = knowhere.GetCurrentVersion()

quant_types = ["SQ4", "RBQ2", "RBQ3", "RBQ4"]
colors = ["red", "blue", "green", "yellow"]
nbits = [4, 2, 3, 4]
# dims = [128, 96, 64, 32]
dims = [768, 640, 512, 384]
# dims = [960, 896, 832, 768, 704, 640, 576, 512, 448, 384, 320]
# dims = [100, 96, 88, 80, 72, 64, 56, 48]

efs = [50, 75, 100, 125, 150, 175, 200, 250, 300]

cfg = {
    "dim": d,
    "metric_type": metric,
    "M": 48,
    "efConstruction": 100,
    "k": topk,
    "ef": topk,
    "graph_build_algo": "vamana",
    "index_algo": "GRAPH",
    "refine_ratio": 1.5
}


def pca_transform(xb, xq, dim):
    d = xb.shape[1]
    pca = faiss.PCAMatrix(d, dim)
    pca.train(xb)
    xb = pca.apply_py(xb)
    xq = pca.apply_py(xq)
    return xb, xq


def rr_transform(xb, xq):
    d = xb.shape[1]
    rr = faiss.RandomRotationMatrix(d, d)
    rr.train(xb)
    xb = rr.apply_py(xb)
    xq = rr.apply_py(xq)
    return xb, xq


def pca_rr_transform(xb, xq, dim):
    xb, xq = pca_transform(xb, xq, dim)
    xb, xq = rr_transform(xb, xq)
    return xb, xq


mp = {}

for dim in dims:
    xb_use, xq_use = pca_rr_transform(xb, xq, dim)
    cfg["dim"] = dim
    cfg["build_quant_type"] = "SQ4"
    idx = knowhere.CreateIndex("HNSW", version)
    idx.Build(knowhere.ArrayToDataSet(xb_use), json.dumps(cfg))
    binset = knowhere.GetBinarySet()
    idx.Serialize(binset)
    index_file = f"test_{datasetname}_{dim}.index"
    knowhere.Dump(binset, index_file)
    for nbit, quant, color in zip(nbits, quant_types, colors):
        name = f"PCA{dim},{quant}"

        cfg["search_quant_type"] = quant
        new_binset = knowhere.GetBinarySet()
        knowhere.Load(new_binset, index_file)
        idx = knowhere.CreateIndex("HNSW", version)
        idx.Deserialize(new_binset, json.dumps(cfg))

        qpss = []
        recalls = []
        for ef in efs:
            cfg["ef"] = ef
            max_qps = 0
            for _ in range(10):
                t0 = time.time()
                ans, _ = idx.Search(knowhere.ArrayToDataSet(xq_use), json.dumps(cfg),
                                    knowhere.GetNullBitSetView())
                t1 = time.time()
                qps = nq / (t1 - t0)
                max_qps = max(max_qps, qps)

            k_dis, k_ids = knowhere.DataSetToArray(ans)

            cnt = 0
            for i in range(nq):
                cnt += np.intersect1d(k_ids[i], gt[i]).size
            recall = cnt / (nq * topk)
            print(f'{name} ef={ef} {recall} {max_qps}')
            if recall > 0.7:
                qpss.append(max_qps)
                recalls.append(recall)
        mp[name] = [qpss, recalls, color]

sns.set_style('whitegrid')
plt.figure(figsize=(16, 12))
for name, (qpss, recalls, color) in mp.items():
    plt.plot(recalls, qpss, label=name, marker='xo'[random.randint(0, 1)])
plt.legend()
plt.title(f'{datasetname} {d}D')
plt.xlabel('Recall@100')
plt.ylabel('QPS')
plt.savefig(f'pca_bench_qps_{datasetname}.png')

with open('fuck.json') as f:
    json.dump(mp, f)
