#!/usr/bin/env python3
import json
import gzip
import numpy as np  # pyright: ignore[reportMissingImports]
import os
import argparse
import faiss  # pyright: ignore[reportMissingImports]

def quantize(v):
	return int(round(v * 10000))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", default="../resources/references.json.gz")
	parser.add_argument("--output-dir", default="output")
	parser.add_argument("--nlist", type=int, default=4096)
	args = parser.parse_args()

	print(f"Loading references from {args.input}...")
	open_fn = gzip.open if args.input.endswith(".gz") else open
	with open_fn(args.input, "rt", encoding="utf-8") as f:
		refs = json.load(f)

	n = len(refs)
	dim = 14
	print(f"Processing {n} vectors...")

	vectors = np.empty((n, dim), dtype=np.float32)
	labels = np.empty(n, dtype=np.uint8)

	for i, r in enumerate(refs):
		vectors[i] = r["vector"]
		labels[i] = 1 if r["label"] == "fraud" else 0
		
	del refs

	# 1. Clustering with FAISS
	print(f"Clustering into {args.nlist} cells...")
	# Using simple KMeans to get centroids
	kmeans = faiss.Kmeans(dim, args.nlist, niter=20, verbose=True)
	kmeans.train(vectors)
	centroids = kmeans.centroids # (nlist, dim)
		
	print("Assigning vectors to clusters...")
	index = faiss.IndexFlatL2(dim)
	index.add(centroids)
	_, assignments = index.search(vectors, 1)
	assignments = assignments.flatten()

	# 2. Group indices by cell
	print("Grouping vectors by cell...")
	cell_indices = [[] for _ in range(args.nlist)]
	for i, cell_id in enumerate(assignments):
		cell_indices[cell_id].append(i)

	# 3. Create reordered arrays and offsets
	print("Reordering vectors and labels...")
	ivf_vectors = np.zeros((n, dim), dtype=np.int16)
	ivf_labels_bits = bytearray((n + 7) // 8)
	ivf_offsets = np.zeros(args.nlist + 1, dtype=np.uint32)

	current_idx = 0
	for cell_id, indices in enumerate(cell_indices):
		ivf_offsets[cell_id] = current_idx
		for idx in indices:
			v = vectors[idx]
			for j in range(dim):
				ivf_vectors[current_idx, j] = quantize(v[j])
			
			# Pack label
			if labels[idx] == 1:
					byte_idx = current_idx // 8
					bit_idx = current_idx % 8
					ivf_labels_bits[byte_idx] |= (1 << bit_idx)
			
			current_idx += 1
		
	ivf_offsets[args.nlist] = n

	# 4. Save files
	os.makedirs(args.output_dir, exist_ok=True)
		
	paths = {
		"centroids": os.path.join(args.output_dir, "centroids.bin"),
		"offsets": os.path.join(args.output_dir, "ivf_offsets.bin"),
		"vectors": os.path.join(args.output_dir, "ivf_vectors.bin"),
		"labels": os.path.join(args.output_dir, "ivf_labels.bin")
	}

	print(f"Saving centroids to {paths['centroids']}...")
	centroids.astype(np.float32).tofile(paths['centroids'])

	print(f"Saving offsets to {paths['offsets']}...")
	ivf_offsets.tofile(paths['offsets'])

	print(f"Saving vectors to {paths['vectors']}...")
	ivf_vectors.tofile(paths['vectors'])

	print(f"Saving labels to {paths['labels']}...")
	with open(paths['labels'], "wb") as f:
		f.write(ivf_labels_bits)

	print("Preprocessing complete!")

if __name__ == "__main__":
	main()