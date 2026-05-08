package main

/*
#cgo LDFLAGS: -lxgboost
#include <xgboost/c_api.h>
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"time"
	"unsafe"
)

// PredictRequest represents a single inference task
type PredictRequest struct {
	features [14]float32
	result   chan float64
}

// Predictor handles XGBoost inference via a single-threaded CGo worker.
type Predictor struct {
	handle C.BoosterHandle
	queue  chan PredictRequest
}

// NewPredictor initializes the XGBoost booster and starts the dedicated worker.
func NewPredictor(modelPath string) (*Predictor, error) {
	var handle C.BoosterHandle
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	res := C.XGBoosterCreate(nil, 0, &handle)
	if res != 0 {
		return nil, fmt.Errorf("failed to create booster")
	}

	res = C.XGBoosterLoadModel(handle, cPath)
	if res != 0 {
		return nil, fmt.Errorf("failed to load model from %s", modelPath)
	}

	// Ensure nthread is 1
	pKey := C.CString("nthread")
	pValue := C.CString("1")
	defer C.free(unsafe.Pointer(pKey))
	defer C.free(unsafe.Pointer(pValue))
	C.XGBoosterSetParam(handle, pKey, pValue)

	p := &Predictor{
		handle: handle,
		queue:  make(chan PredictRequest, 1024),
	}

	// Start the dedicated CGo worker
	go p.worker()

	return p, nil
}

// worker processes predictions sequentially. This is the ULTRA-STABLE path.
func (p *Predictor) worker() {
	for req := range p.queue {
		var dmatrix C.DMatrixHandle

		// Create DMatrix from a single row
		// XGDMatrixCreateFromMat(const float* data, bst_ulong nrow, bst_ulong ncol, float missing, DMatrixHandle* out)
		res := C.XGDMatrixCreateFromMat(
			(*C.float)(unsafe.Pointer(&req.features[0])),
			C.bst_ulong(1),
			C.bst_ulong(14),
			C.float(0.0), // We assume 0.0 is not "missing" here, or use NaN if needed
			&dmatrix,
		)

		if res != 0 {
			req.result <- 0.0
			continue
		}

		var outLen C.bst_ulong
		var outPtr *C.float

		// Predict
		// XGBoosterPredict(BoosterHandle handle, DMatrixHandle dmat, int option_mask, unsigned ntree_limit, int training, bst_ulong* len, const float** out_result)
		res = C.XGBoosterPredict(
			p.handle,
			dmatrix,
			0, // option_mask
			0, // ntree_limit
			0, // training (0 = false)
			&outLen,
			&outPtr,
		)

		if res != 0 || outLen == 0 {
			C.XGDMatrixFree(dmatrix)
			req.result <- 0.0
			continue
		}

		score := float64(*outPtr)

		// Clean up DMatrix immediately
		C.XGDMatrixFree(dmatrix)

		// Clamp to [0, 1]
		if score < 0 {
			score = 0
		}
		if score > 1 {
			score = 1
		}

		req.result <- score
	}
}

// Predict submits a task to the worker and waits for the result.
func (p *Predictor) Predict(features [14]float32) (float64, bool) {
	resChan := make(chan float64, 1)
	req := PredictRequest{
		features: features,
		result:   resChan,
	}

	select {
	case p.queue <- req:
		select {
		case res := <-resChan:
			return res, true
		case <-time.After(50 * time.Millisecond):
			return 0.0, true
		}
	default:
		return 0.0, true
	}
}

// Close releases the booster handle.
func (p *Predictor) Close() {
	close(p.queue)
	C.XGBoosterFree(p.handle)
}
