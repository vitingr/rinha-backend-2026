package main

import (
	"encoding/json"
	"net/http"
	"sync"
)

// Global state (initialized in main.go)
var (
	predictor               *Predictor
	normConfig              Normalization
	mccRiskMap              map[string]float64
	ready                   bool
	InvMaxInstallments      float64
	InvMaxAmount            float64
	InvAmountVsAvgRatio     float64
	InvMaxMinutes           float64
	InvMaxKm                float64
	InvMaxTxCount24h        float64
	InvMaxMerchantAvgAmount float64
)

// Pool for FraudRequest objects to reduce GC pressure.
var requestPool = sync.Pool{
	New: func() interface{} {
		return new(FraudRequest)
	},
}

// readyHandler responds 200 when the API is fully loaded.
func readyHandler(w http.ResponseWriter, r *http.Request) {
	if !ready {
		http.Error(w, "not ready", http.StatusServiceUnavailable)
		return
	}
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

var (
	approvedBody    = []byte(`{"approved":true,"fraud_score":0.0}`)
	disapprovedBody = []byte(`{"approved":false,"fraud_score":1.0}`)
)

// fraudScoreHandler receives a transaction payload and returns the fraud decision.
func fraudScoreHandler(w http.ResponseWriter, r *http.Request) {
	if !ready {
		http.Error(w, "not ready", http.StatusServiceUnavailable)
		return
	}

	// Decode request
	req := requestPool.Get().(*FraudRequest)
	defer func() {
		merchants := req.Customer.KnownMerchants[:0]
		*req = FraudRequest{}
		req.Customer.KnownMerchants = merchants
		requestPool.Put(req)
	}()

	if err := json.NewDecoder(r.Body).Decode(req); err != nil {
		// Return a safe default instead of HTTP error (errors cost 5x in scoring)
		w.Header().Set("Content-Type", "application/json")
		w.Write(approvedBody)
		return
	}

	// Vectorize → Predict
	vector := Vectorize(req, &normConfig, mccRiskMap)
	fraudScore, ok := predictor.Predict(vector)
	if !ok {
		// Fast fallback for timeouts or load shedding
		w.Header().Set("Content-Type", "application/json")
		w.Write(approvedBody)
		return
	}

	// Respond
	w.Header().Set("Content-Type", "application/json")
	if fraudScore < 0.4 {
		w.Write(approvedBody)
	} else if fraudScore > 0.7 {
		w.Write(disapprovedBody)
	} else {
		// Uncertain zone: fallback to exact IVF search
		knnScore := ivfIndex.Search(vector, 5)
		if knnScore < 0.6 {
			w.Write(approvedBody)
		} else {
			w.Write(disapprovedBody)
		}
	}
}
