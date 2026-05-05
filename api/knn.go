package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

const (
	nlist  = 4096
	nprobe = 16
	dim    = 14
)

// IVFIndex holds the clustered reference vectors and their labels.
type IVFIndex struct {
	centroids []float32 // nlist * dim
	offsets   []uint32  // nlist + 1
	vectors   []int16   // 3M * dim
	labels    []byte    // 3M bits
	n         int
}

// LoadIVFIndex loads the preprocessed IVF binary files.
func LoadIVFIndex(centroidsPath, offsetsPath, vectorsPath, labelsPath string) (*IVFIndex, error) {
	cData, err := os.ReadFile(centroidsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read centroids: %w", err)
	}
	centroids := make([]float32, len(cData)/4)
	for i := 0; i < len(centroids); i++ {
		centroids[i] = math.Float32frombits(binary.LittleEndian.Uint32(cData[i*4 : i*4+4]))
	}

	oData, err := os.ReadFile(offsetsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read offsets: %w", err)
	}
	offsets := make([]uint32, len(oData)/4)
	for i := 0; i < len(offsets); i++ {
		offsets[i] = binary.LittleEndian.Uint32(oData[i*4 : i*4+4])
	}

	vecData, err := os.ReadFile(vectorsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read vectors: %w", err)
	}
	vectors := make([]int16, len(vecData)/2)
	for i := 0; i < len(vectors); i++ {
		vectors[i] = int16(binary.LittleEndian.Uint16(vecData[i*2 : i*2+2]))
	}

	labelData, err := os.ReadFile(labelsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read labels: %w", err)
	}

	return &IVFIndex{
		centroids: centroids,
		offsets:   offsets,
		vectors:   vectors,
		labels:    labelData,
		n:         len(vectors) / dim,
	}, nil
}

// Search performs an IVF search and returns the fraud score.
func (idx *IVFIndex) Search(query [14]float32, k int) float64 {
	// 1. Find top nprobe centroids
	var topCells [nprobe]int
	var topCellDists [nprobe]float32
	for i := range topCellDists {
		topCellDists[i] = math.MaxFloat32
	}

	for i := 0; i < nlist; i++ {
		var d float32
		offset := i * dim
		for j := 0; j < dim; j++ {
			diff := query[j] - idx.centroids[offset+j]
			d += diff * diff
		}

		if d < topCellDists[nprobe-1] {
			pos := nprobe - 1
			for pos > 0 && d < topCellDists[pos-1] {
				topCellDists[pos] = topCellDists[pos-1]
				topCells[pos] = topCells[pos-1]
				pos--
			}
			topCellDists[pos] = d
			topCells[pos] = i
		}
	}

	// 2. Quantize query for int16 distance
	var q [dim]int16
	for i, v := range query {
		if v < 0 {
			q[i] = int16(v*10000.0 - 0.5)
		} else {
			q[i] = int16(v*10000.0 + 0.5)
		}
	}

	// 3. Search in top cells
	// Use uint64 for dist because 14 * (20000^2) = 5.6B which exceeds uint32 (4.29B)
	var topDist [5]uint64
	var topIdx [5]int
	for i := 0; i < 5; i++ {
		topDist[i] = math.MaxUint64
	}

	for _, cellID := range topCells {
		start := int(idx.offsets[cellID])
		end := int(idx.offsets[cellID+1])

		for i := start; i < end; i++ {
			offset := i * dim
			var dist uint64

			// Manual unrolling for 14 dimensions
			v0 := int32(idx.vectors[offset]) - int32(q[0])
			v1 := int32(idx.vectors[offset+1]) - int32(q[1])
			v2 := int32(idx.vectors[offset+2]) - int32(q[2])
			v3 := int32(idx.vectors[offset+3]) - int32(q[3])
			v4 := int32(idx.vectors[offset+4]) - int32(q[4])
			v5 := int32(idx.vectors[offset+5]) - int32(q[5])
			v6 := int32(idx.vectors[offset+6]) - int32(q[6])
			v7 := int32(idx.vectors[offset+7]) - int32(q[7])
			dist = uint64(v0*v0) + uint64(v1*v1) + uint64(v2*v2) + uint64(v3*v3) + uint64(v4*v4) + uint64(v5*v5) + uint64(v6*v6) + uint64(v7*v7)

			if dist >= topDist[4] {
				continue
			}

			v8 := int32(idx.vectors[offset+8]) - int32(q[8])
			v9 := int32(idx.vectors[offset+9]) - int32(q[9])
			v10 := int32(idx.vectors[offset+10]) - int32(q[10])
			v11 := int32(idx.vectors[offset+11]) - int32(q[11])
			v12 := int32(idx.vectors[offset+12]) - int32(q[12])
			v13 := int32(idx.vectors[offset+13]) - int32(q[13])

			dist += uint64(v8*v8) + uint64(v9*v9) + uint64(v10*v10) + uint64(v11*v11) + uint64(v12*v12) + uint64(v13*v13)

			if dist < topDist[4] {
				pos := 4
				for pos > 0 && dist < topDist[pos-1] {
					topDist[pos] = topDist[pos-1]
					topIdx[pos] = topIdx[pos-1]
					pos--
				}
				topDist[pos] = dist
				topIdx[pos] = i
			}
		}
	}

	// 4. Score
	fraudCount := 0
	for i := 0; i < 5; i++ {
		idxVal := topIdx[i]
		if idxVal == 0 && topDist[i] == math.MaxUint64 {
			continue
		}
		byteIdx := idxVal / 8
		bitIdx := uint(idxVal % 8)
		if (idx.labels[byteIdx] & (1 << bitIdx)) != 0 {
			fraudCount++
		}
	}

	return float64(fraudCount) / 5.0
}