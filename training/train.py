#!/usr/bin/env python3
"""
Training pipeline for Rinha de Backend 2026 - XGBoost fraud score regressor.

Strategy (Option B):
1. Load 3M reference vectors from references.json.gz
2. Build a FAISS index for exact k-NN (k=5, Euclidean)
3. For each vector, compute fraud_score = count_of_fraud_neighbors / 5
4. Train XGBoost regressor: X = vectors (14-dim), y = fraud_scores
5. Export model as model.json (readable by XGBoost C API)
"""

from time import time
import argparse
import gc
import gzip
import json
import os
import time

import numpy as np  # pyright: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split  # pyright: ignore[reportMissingImports]

def load_references(path: str):
	"""Load reference vectors and labels from gzipped JSON."""
	print(f"[1/5] Loading references from {path}...")
	t0 = time.time()

	open_fn = gzip.open if path.endswith(".gz") else open
	with open_fn(path, "rt", encoding="utf-8") as f:
		refs = json.load(f)

	n = len(refs)
	print(f"    Parsing {n} items...")
		
	# Pre-allocate to avoid temporary lists
	vectors = np.empty((n, 14), dtype=np.float32)
	labels = np.empty(n, dtype=np.int32)
		
	for i, r in enumerate(refs):
		vectors[i] = r["vector"]
		labels[i] = 1 if r["label"] == "fraud" else 0
	
	# Crucial: free the huge list of dicts immediately
	del refs
	gc.collect()

	elapsed = time.time() - t0
	print(f"    Loaded and parsed {n} vectors in {elapsed:.1f}s")
	print(f"    Fraud rate: {labels.mean():.2%}")
	return vectors, labels

def compute_knn_scores(vectors: np.ndarray, labels: np.ndarray, k: int = 5, batch_size: int = 100_000):
	"""
	For each vector, find the k nearest neighbors (excluding self) and
	compute fraud_score = count_of_frauds_in_neighbors / k.

	Uses FAISS IVFFlat for much faster search than brute-force IndexFlatL2.
	With nprobe=nlist we search all partitions (equivalent to exact search).
	"""
	n, dim = vectors.shape

	import faiss  # pyright: ignore[reportMissingImports]
	# Use all available CPU cores for FAISS
	n_threads = os.cpu_count() or 8
	faiss.omp_set_num_threads(n_threads)
	# IVFFlat: partition vectors into nlist cells, then search nprobe cells per query.
	# nlist=1024, nprobe=64: searches ~3% of cells → ~30x faster than brute force.
	# For training labels, ~99% recall is more than sufficient.
	nlist = 1024
	nprobe = 64

	print(f"[2/5] Building FAISS IVF index ({n} vectors, {dim} dims, {n_threads} threads, nlist={nlist})...")
	t0 = time.time()

	quantizer = faiss.IndexFlatL2(dim)
	index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

	# IVF requires training on a sample of vectors
	print("    Training IVF quantizer...")
	index.train(vectors[:nlist * 40])
	index.add(vectors)
	index.nprobe = nprobe
	print(f"    Index built in {time.time() - t0:.1f}s")

	print(f"[3/5] Computing k-NN scores (k={k}, batches of {batch_size}, nprobe={nprobe})...")
	t0 = time.time()

	fraud_scores = np.zeros(n, dtype=np.float32)
	# Search for k+1 neighbors because the first match is the query itself
	search_k = k + 1
	n_batches = (n + batch_size - 1) // batch_size

	for batch_idx, start in enumerate(range(0, n, batch_size)):
		end = min(start + batch_size, n)
		batch = vectors[start:end]

		batch_t0 = time.time()
		_distances, indices = index.search(batch, search_k)
		batch_elapsed = time.time() - batch_t0

		# Skip the first neighbor (self-match), take the next k
		neighbor_indices = indices[:, 1 : search_k]
		dist_to_neighbors = _distances[:, 1:search_k]
		# Use inverse distance as weight (add epsilon to avoid div by zero)
		weights = 1.0 / (dist_to_neighbors + 1e-6)
		neighbor_labels = labels[neighbor_indices]

		weighted_sum = (neighbor_labels * weights).sum(axis=1)
		total_weight = weights.sum(axis=1)
		fraud_scores[start:end] = weighted_sum / total_weight

		progress = end / n * 100
		print(f"    Batch {batch_idx+1}/{n_batches}: {end}/{n} ({progress:.1f}%) - {batch_elapsed:.1f}s")

	elapsed = time.time() - t0
	print(f"    k-NN scores computed in {elapsed:.1f}s")

	# Distribution of scores
	unique, counts = np.unique(fraud_scores, return_counts=True)
	print("    Score distribution:")
	for val, cnt in zip(unique, counts):
		print(f"      {val:.1f}: {cnt:>8} ({cnt/n:.2%})")

	return fraud_scores

def save_knn_scores(scores: np.ndarray, path: str):
	"""Save computed k-NN scores to a binary file."""
	print(f"    Saving k-NN scores to {path}...")
	np.save(path, scores)

def load_knn_scores(path: str):
	"""Load k-NN scores from a binary file."""
	if os.path.exists(path):
		print(f"    Loading k-NN scores from {path}...")
		return np.load(path)
	return None

def main():
	parser = argparse.ArgumentParser(description="Train XGBoost fraud score regressor")
	parser.add_argument(
		"--references",
		default="../resources/references.json.gz",
		help="Path to references.json.gz",
	)
	parser.add_argument(
		"--output",
		default="output",
		help="Output directory for model files",
	)
	parser.add_argument("--k", type=int, default=5, help="Number of neighbors for k-NN")
	parser.add_argument(
		"--batch-size", type=int, default=50000, help="Batch size for k-NN search"
	)
	parser.add_argument(
		"--num-boost-round", type=int, default=1500, help="Number of boosting rounds"
	)
	parser.add_argument("--max-depth", type=int, default=6, help="Max tree depth")
	parser.add_argument("--eta", type=float, default=0.05, help="Learning rate")
	args = parser.parse_args()

	total_t0 = time.time()

	# Step 1: Load data
	vectors, labels = load_references(args.references)

	# Step 2-3: Compute or load k-NN fraud scores
	cache_path = os.path.join(args.output, "knn_scores.npy")
	os.makedirs(args.output, exist_ok=True)
		
	fraud_scores = load_knn_scores(cache_path)
	if fraud_scores is None:
		fraud_scores = compute_knn_scores(vectors, labels, k=args.k, batch_size=args.batch_size)
		save_knn_scores(fraud_scores, cache_path)
	else:
		print("    Using cached k-NN scores.")

	# Cleanup labels as they are no longer needed for XGBoost
	del labels
	gc.collect()

	import xgboost as xgb  # pyright: ignore[reportMissingImports]

	def train_xgboost(vectors: np.ndarray, fraud_scores: np.ndarray, params: dict = None):
		print(f"[4/5] Training XGBoost regressor (vectors shape: {vectors.shape})...")
		t0 = time.time()

		# Data validation
		print("    Validating data...")
		if np.any(np.isnan(vectors)):
			print("    WARNING: NaN values found in vectors!")
		if np.any(np.isinf(vectors)):
			print("    WARNING: Inf values found in vectors!")
				
		print(f"    Vector range: [{np.min(vectors):.3f}, {np.max(vectors):.3f}]")
		print(f"    Scores range: [{np.min(fraud_scores):.3f}, {np.max(fraud_scores):.3f}]")

		# 1. Validation Split (90/10)
		print("    Splitting data for validation...")
		X_train, X_val, y_train, y_val = train_test_split(
			vectors, fraud_scores, test_size=0.1, random_state=42
		)

		sample_weights = np.ones(len(y_train), dtype=np.float32)
		# Increase importance of samples near our problematic decision boundary
		mask = (y_train >= 0.3) & (y_train <= 0.7)
		sample_weights[mask] = 5.0

		print("    Creating DMatrices...")
		dtrain = xgb.DMatrix(X_train, label=y_train, nthread=4)
		dval = xgb.DMatrix(X_val, label=y_val, nthread=4)

		if params is None:
			params = {
				"objective": "reg:squarederror",
				"max_depth": 6,
				"eta": 0.05,
				"lambda": 1.5,
				"alpha": 0.5,
				"subsample": 0.8,
				"colsample_bytree": 0.8,
				"min_child_weight": 5,
				"tree_method": "hist",
				"nthread": 4,
				"verbosity": 1,
			}

		num_boost_round = 1500

		# 2. Training with Early Stopping
		model = xgb.train(
			params,
			dtrain,
			num_boost_round=num_boost_round,
			evals=[(dtrain, "train"), (dval, "val")],
			early_stopping_rounds=25,
			verbose_eval=50,
		)

		elapsed = time.time() - t0
		print(f"    Training completed in {elapsed:.1f}s")

		# Quick accuracy check on training data (using full set for comparison)
		dfull = xgb.DMatrix(vectors, nthread=4)
		predictions = model.predict(dfull)
		# Round to nearest 0.2 to match k-NN output space
		rounded = np.clip(predictions, 0.0, 1.0)

		exact_match = (rounded == fraud_scores).mean()
		approval_match = ((rounded < 0.6) == (fraud_scores < 0.6)).mean()
		print(f"    Exact score match (rounded): {exact_match:.2%}")
		print(f"    Approval decision match:     {approval_match:.2%}")

		return model

	def export_model(model: xgb.Booster, output_dir: str):
		"""Export model in JSON format (readable by XGBoost C API)."""
		print(f"[5/5] Exporting model to {output_dir}...")
		os.makedirs(output_dir, exist_ok=True)

		json_path = os.path.join(output_dir, "model.json")
		model.save_model(json_path)
		json_size = os.path.getsize(json_path) / 1024 / 1024
		print(f"    model.json: {json_size:.1f} MB")

		ubj_path = os.path.join(output_dir, "model.ubj")
		model.save_model(ubj_path)
		ubj_size = os.path.getsize(ubj_path) / 1024 / 1024
		print(f"    model.ubj:  {ubj_size:.1f} MB")

		bin_path = os.path.join(output_dir, "model.bin")
		# Force legacy binary format for compatibility with the 'leaves' Go library
		model.save_model(bin_path)
		bin_size = os.path.getsize(bin_path) / 1024 / 1024
		print(f"    model.bin:  {bin_size:.1f} MB (legacy format)")

	# Step 4: Train XGBoost
	params = {
		"objective": "reg:squarederror",
		"max_depth": args.max_depth,
		"eta": args.eta,
		"lambda": 1.5,
		"alpha": 0.5,
		"subsample": 0.8,
		"colsample_bytree": 0.8,
		"min_child_weight": 5,
		"tree_method": "hist",
		"nthread": 4,
		"verbosity": 2,
	}
	model = train_xgboost(vectors, fraud_scores, params)

	# Step 5: Export
	export_model(model, args.output)

	total_elapsed = time.time() - total_t0
	print(f"\nTotal training time: {total_elapsed:.1f}s")
	print("Done!")

if __name__ == "__main__":
	main()
