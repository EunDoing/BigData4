"""
MovieLens 1M 스트리밍 알고리즘 실험
- Bloom Filter + Count-Min Sketch
- 스트림 1회 통과로 모든 파라미터 동시 실험
"""

import mmh3
import math
import time
import tracemalloc
import sys
import json
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "/home/claude/ml-1m/ratings.dat"

class BloomFilter:
    def __init__(self, m, k):
        self.m = m
        self.k = k
        self.bits = bytearray(math.ceil(m / 8))
        self.fp = 0
        self.fn = 0
        self.queries = 0

    def _hashes(self, item):
        b = str(item).encode()
        return [mmh3.hash(b, seed=i) % self.m for i in range(self.k)]

    def add(self, item):
        for h in self._hashes(item):
            self.bits[h // 8] |= (1 << (h % 8))

    def __contains__(self, item):
        return all((self.bits[h // 8] >> (h % 8)) & 1 for h in self._hashes(item))

    def memory_bytes(self):
        return len(self.bits)

class CountMinSketch:
    def __init__(self, w, d):
        self.w = w
        self.d = d
        self.table = np.zeros((d, w), dtype=np.int32)

    def _hashes(self, item):
        b = str(item).encode()
        return [mmh3.hash(b, seed=i) % self.w for i in range(self.d)]

    def add(self, item):
        for i, h in enumerate(self._hashes(item)):
            self.table[i][h] += 1

    def query(self, item):
        return min(self.table[i][h] for i, h in enumerate(self._hashes(item)))

    def memory_bytes(self):
        return self.table.nbytes

def stream_ratings(path):
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) == 4:
                yield int(parts[0]), int(parts[1])

# BF 파라미터 설정
BF_CONFIGS = [
    (100_000,  3),
    (500_000,  5),
    (1_000_000, 7),
    (5_000_000, 10),
]

# CMS 파라미터 설정
CMS_CONFIGS = [
    (1000, 3),
    (5000, 4),
    (10000, 5),
    (50000, 7),
]

# 구조체 초기화
bfs = [BloomFilter(m, k) for m, k in BF_CONFIGS]
cmss = [CountMinSketch(w, d) for w, d in CMS_CONFIGS]

# Ground Truth
seen_users_exact = set()
movie_count_exact = defaultdict(int)

print("스트림 처리 시작...")
tracemalloc.start()
t_start = time.time()
total = 0

for user_id, movie_id in stream_ratings(DATA_PATH):
    already_seen = user_id in seen_users_exact

    for bf in bfs:
        bf_says_seen = user_id in bf
        if bf_says_seen and not already_seen:
            bf.fp += 1
        if not bf_says_seen and already_seen:
            bf.fn += 1
        bf.queries += 1
        bf.add(user_id)

    for cms in cmss:
        cms.add(movie_id)

    if not already_seen:
        seen_users_exact.add(user_id)
    movie_count_exact[movie_id] += 1
    total += 1

t_end = time.time()
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

elapsed = t_end - t_start
print(f"완료: {total:,}건, {elapsed:.2f}s, throughput={total/elapsed:,.0f}/s")
print(f"고유 사용자: {len(seen_users_exact)}, 고유 영화: {len(movie_count_exact)}")

# ── CMS 오차 계산 ──
movie_ids = list(movie_count_exact.keys())
cms_stats = []
for cms in cmss:
    errors = [abs(cms.query(mid) - movie_count_exact[mid]) for mid in movie_ids]
    rel_errors = [abs(cms.query(mid) - movie_count_exact[mid]) / movie_count_exact[mid] for mid in movie_ids]
    cms_stats.append({
        "w": cms.w, "d": cms.d,
        "mean_rel_error": float(np.mean(rel_errors)),
        "mean_abs_error": float(np.mean(errors)),
        "max_abs_error": float(np.max(errors)),
        "memory_kb": cms.memory_bytes() / 1024,
    })

bf_stats = []
for bf in bfs:
    bf_stats.append({
        "m": bf.m, "k": bf.k,
        "fpr": bf.fp / bf.queries,
        "fn": bf.fn,
        "memory_kb": bf.memory_bytes() / 1024,
    })

print("\n── Bloom Filter 결과 ──")
for s in bf_stats:
    print(f"m={s['m']:>9,}, k={s['k']}: FPR={s['fpr']:.6f}, mem={s['memory_kb']:.1f}KB")

print("\n── Count-Min Sketch 결과 ──")
for s in cms_stats:
    print(f"w={s['w']:>6,}, d={s['d']}: mean_rel_err={s['mean_rel_error']:.6f}, mem={s['memory_kb']:.1f}KB")

# ── 그래프 ──
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Streaming Algorithm Experiment - MovieLens 1M (1,000,209 ratings)", fontsize=13)

# BF: FPR
ax = axes[0][0]
ms = [s['m'] for s in bf_stats]
fprs = [s['fpr'] for s in bf_stats]
ax.plot(ms, fprs, 'o-', color='steelblue', linewidth=2, markersize=8)
for x, y in zip(ms, fprs):
    ax.annotate(f"{y:.4%}", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)
ax.set_title("Bloom Filter: FPR vs Bit Array Size (m)")
ax.set_xlabel("m (bits)")
ax.set_ylabel("False Positive Rate")
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# BF: Memory
ax = axes[0][1]
mems = [s['memory_kb'] for s in bf_stats]
bars = ax.bar([f"{m//1000}K" for m in ms], mems, color='steelblue', alpha=0.8)
ax.set_title("Bloom Filter: Memory vs m")
ax.set_xlabel("m")
ax.set_ylabel("Memory (KB)")
ax.grid(True, axis='y', alpha=0.3)
for bar, v in zip(bars, mems):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f"{v:.0f}KB", ha='center', fontsize=8)

# BF: FPR vs Memory (trade-off)
ax = axes[0][2]
ax.plot(mems, fprs, 'D-', color='navy', linewidth=2, markersize=8)
for x, y, m in zip(mems, fprs, ms):
    ax.annotate(f"m={m//1000}K", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_title("Bloom Filter: FPR vs Memory (Trade-off)")
ax.set_xlabel("Memory (KB)")
ax.set_ylabel("False Positive Rate")
ax.grid(True, alpha=0.3)

# CMS: Relative Error
ax = axes[1][0]
ws = [s['w'] for s in cms_stats]
rel_errs = [s['mean_rel_error'] for s in cms_stats]
ax.plot(ws, rel_errs, 's-', color='tomato', linewidth=2, markersize=8)
for x, y in zip(ws, rel_errs):
    ax.annotate(f"{y:.4%}", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)
ax.set_title("Count-Min Sketch: Mean Relative Error vs Width (w)")
ax.set_xlabel("w (columns)")
ax.set_ylabel("Mean Relative Error")
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# CMS: Memory
ax = axes[1][1]
cms_mems = [s['memory_kb'] for s in cms_stats]
bars = ax.bar([f"w={w}" for w in ws], cms_mems, color='tomato', alpha=0.8)
ax.set_title("Count-Min Sketch: Memory vs w")
ax.set_xlabel("w")
ax.set_ylabel("Memory (KB)")
ax.grid(True, axis='y', alpha=0.3)
for bar, v in zip(bars, cms_mems):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{v:.1f}KB", ha='center', fontsize=8)

# CMS: Error vs Memory (trade-off)
ax = axes[1][2]
ax.plot(cms_mems, rel_errs, '^-', color='darkred', linewidth=2, markersize=8)
for x, y, w in zip(cms_mems, rel_errs, ws):
    ax.annotate(f"w={w}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_title("Count-Min Sketch: Error vs Memory (Trade-off)")
ax.set_xlabel("Memory (KB)")
ax.set_ylabel("Mean Relative Error")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/results.png", dpi=150, bbox_inches='tight')

# 결과 저장
results = {
    "summary": {
        "total_ratings": total,
        "elapsed_sec": elapsed,
        "throughput_per_sec": total / elapsed,
        "peak_memory_mb": peak_mem / 1024 / 1024,
        "unique_users": len(seen_users_exact),
        "unique_movies": len(movie_count_exact),
    },
    "bloom_filter": bf_stats,
    "count_min_sketch": cms_stats,
}
with open("/home/claude/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n완료: results.png, results.json")
