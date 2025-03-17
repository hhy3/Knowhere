import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import faiss
import knowhere
import json


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


datasetname = 'glove'
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

quant_types = ["SQ4", "RBQ1", "RBQ2", "RBQ3", "RBQ4"]
colors = ["red", "blue", "green", "yellow", "purple"]
nbits = [4, 1, 2, 3, 4]
# dims = [128, 112, 96, 80, 64]
# dims = [768, 704, 640, 576, 512, 448, 384, 320, 256]
# dims = [960, 896, 832, 768, 704, 640, 576, 512, 448, 384, 320]
dims = [100, 96, 88, 80, 72, 64, 56, 48]

cfg = {
    "dim": d,
    "metric_type": metric,
    "k": topk,
    "index_algo": "QUANTBF",
    "refine_ratio": 1.0
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
    for nbit, quant, color in zip(nbits, quant_types, colors):
        bpw = dim / xb.shape[1] * nbit
        name = f"PCA{dim},{quant}"

        cfg["dim"] = dim
        cfg["build_quant_type"] = quant
        cfg["search_quant_type"] = quant
        idx = knowhere.CreateIndex("HNSW", version)
        idx.Build(knowhere.ArrayToDataSet(xb_use), json.dumps(cfg))

        ans, _ = idx.Search(knowhere.ArrayToDataSet(xq_use), json.dumps(cfg),
                            knowhere.GetNullBitSetView())

        k_dis, k_ids = knowhere.DataSetToArray(ans)

        cnt = 0
        for i in range(nq):
            cnt += np.intersect1d(k_ids[i], gt[i]).size
        recall = cnt / (nq * topk)
        print(name, recall)
        mp[name] = [bpw, recall, color]

sns.set_style('whitegrid')
plt.figure(figsize=(16, 12))
vals = list(mp.values())
x, y, c = [_[0] for _ in vals], [_[1] for _ in vals], [_[2] for _ in vals]
print(x, y)
plt.scatter(x, y, marker='o', c=c)
for i, (xi, yi, label, color) in enumerate(zip(x, y, list(mp.keys()), c)):
    plt.text(xi-0.02, yi-0.005, label, rotation=-
             45, fontsize=9, ha='left', va='bottom')
plt.xlabel('bpw')
plt.ylabel('Recall@100')
plt.title(f'{datasetname} {d}D')
plt.savefig(f'pca_bench_{datasetname}.png')
