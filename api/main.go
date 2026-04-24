package main

import (
	"compress/gzip"
	"encoding/json"
	"log"
	"math"
	"net/http"
	"os"
	"runtime"
	"sync"
	"time"
)

const (
	dims                 = 14
	kNeighbors           = 5
	fraudThreshold       = 0.6
	maxAmount            = float32(10000)
	maxInstallments      = float32(12)
	amountVsAvgRatio     = float32(10)
	maxMinutes           = float32(1440)
	maxKm                = float32(1000)
	maxTxCount24h        = float32(20)
	maxMerchantAvgAmount = float32(10000)
)

// MCC risk table — hardcoded from mcc_risk.json, zero I/O at startup
var mccRisk = map[string]float32{
	"5411": 0.15,
	"5812": 0.30,
	"5912": 0.20,
	"5944": 0.45,
	"7801": 0.80,
	"7802": 0.75,
	"7995": 0.85,
	"4511": 0.35,
	"5311": 0.25,
	"5999": 0.50,
}

// Dataset
// Layout: refVecs[i*dims : i*dims+dims] = 14-dim float32 vector i
//         refFraud[i] == 1 means fraud
//
// 100k * 14 * 4 = 5.6 MB (vectors) + 100 KB (labels) = ~5.7 MB total
// Fits entirely in L3 cache → after warmup each KNN is pure cache hits
var (
	refVecs  []float32
	refFraud []uint8
	refN     int

	readyOnce sync.Once
	readyCh   = make(chan struct{})
)

// Precomputed responses — all 6 possible outcomes, built once at startup.
// fraud_score ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} → index by fraudCount (0-5)
var httpResp [kNeighbors + 1][]byte

func buildResponses() {
	scoreStr := [kNeighbors + 1]string{"0.0", "0.2", "0.4", "0.6", "0.8", "1.0"}
	for i := 0; i <= kNeighbors; i++ {
		approved := "true"
		if float64(i)/float64(kNeighbors) >= fraudThreshold {
			approved = "false"
		}
		httpResp[i] = []byte(`{"approved":` + approved + `,"fraud_score":` + scoreStr[i] + `}`)
	}
}

// KNN — brute-force linear scan with a fixed-size max-heap
//
// Benchmark results (Intel Xeon 2.8GHz, 2 hardware threads, 100k vectors):
//   1 goroutine: ~1.07 ms/op, 0 allocs
//   2 goroutines: ~1.25 ms/op, 3 allocs  ← goroutine overhead > gain
//   4 goroutines: ~1.37 ms/op, 5 allocs
//
// Conclusion: stay single-threaded. The 5.6 MB dataset warms into L3 and
// sequential scan is the fastest memory access pattern. No sync overhead.

// heap5 is a max-heap capped at k=5 elements. Lives on the goroutine stack
// (stack-allocated, zero escape to heap) because it never crosses function
// boundaries by pointer to the outside.
type heap5 struct {
	dist [kNeighbors]float32
	idx  [kNeighbors]int32
	size int32
	max  float32
}

func (h *heap5) push(d float32, i int32) {
	if h.size < kNeighbors {
		j := h.size
		h.dist[j] = d
		h.idx[j] = i
		h.size++
		if h.size == kNeighbors {
			h.max = maxOf5(h.dist)
		}
		return
	}
	// Fast path: reject anything >= current worst (most iterations)
	if d >= h.max {
		return
	}
	// Replace the current worst element
	worst := int32(0)
	for j := int32(1); j < kNeighbors; j++ {
		if h.dist[j] > h.dist[worst] {
			worst = j
		}
	}
	h.dist[worst] = d
	h.idx[worst] = i
	h.max = maxOf5(h.dist)
}

func maxOf5(d [kNeighbors]float32) float32 {
	m := d[0]
	if d[1] > m {
		m = d[1]
	}
	if d[2] > m {
		m = d[2]
	}
	if d[3] > m {
		m = d[3]
	}
	if d[4] > m {
		m = d[4]
	}
	return m
}

func (h *heap5) countFraud() int {
	n := 0
	for j := int32(0); j < h.size; j++ {
		n += int(refFraud[h.idx[j]])
	}
	return n
}

// euclidSq returns the squared Euclidean distance between q and vecs[off:off+14].
// Manually unrolled for 14 dimensions — the Go compiler auto-vectorizes to SSE/AVX.
//
//go:nosplit
func euclidSq(q *[dims]float32, vecs []float32, off int) float32 {
	v := vecs[off : off+dims : off+dims]
	d0 := q[0] - v[0]
	d1 := q[1] - v[1]
	d2 := q[2] - v[2]
	d3 := q[3] - v[3]
	d4 := q[4] - v[4]
	d5 := q[5] - v[5]
	d6 := q[6] - v[6]
	d7 := q[7] - v[7]
	d8 := q[8] - v[8]
	d9 := q[9] - v[9]
	d10 := q[10] - v[10]
	d11 := q[11] - v[11]
	d12 := q[12] - v[12]
	d13 := q[13] - v[13]
	return d0*d0 + d1*d1 + d2*d2 + d3*d3 +
		d4*d4 + d5*d5 + d6*d6 + d7*d7 +
		d8*d8 + d9*d9 + d10*d10 + d11*d11 +
		d12*d12 + d13*d13
}

// knn scans all reference vectors and returns the number of fraud labels
// among the 5 nearest neighbors. Stack-allocated heap, zero dynamic allocations.
func knn(query *[dims]float32) int {
	var h heap5
	h.max = math.MaxFloat32

	vecs := refVecs
	n := refN
	stride := dims

	for i := 0; i < n; i++ {
		d := euclidSq(query, vecs, i*stride)
		h.push(d, int32(i))
	}
	return h.countFraud()
}

// Vectorization — transforms TxRequest → [14]float32, all on the stack
func clamp(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func vectorize(req *TxRequest, out *[dims]float32) {
	// dim 0: normalized transaction amount
	out[0] = clamp(float32(req.Transaction.Amount) / maxAmount)

	// dim 1: normalized installment count
	out[1] = clamp(float32(req.Transaction.Installments) / maxInstallments)

	// dim 2: amount relative to customer's historical average
	if req.Customer.AvgAmount > 0 {
		out[2] = clamp((float32(req.Transaction.Amount) / float32(req.Customer.AvgAmount)) / amountVsAvgRatio)
	} else {
		out[2] = 1.0
	}

	// dim 3: hour of day in UTC, normalized 0-1
	out[3] = float32(parseHour(req.Transaction.RequestedAt)) / 23.0

	// dim 4: day of week (Mon=0, Sun=6), normalized 0-1
	out[4] = float32(parseDOW(req.Transaction.RequestedAt)) / 6.0

	// dims 5-6: time & distance since last transaction (-1 sentinel if null)
	if req.LastTransaction == nil {
		out[5] = -1
		out[6] = -1
	} else {
		mins := minutesBetween(req.LastTransaction.Timestamp, req.Transaction.RequestedAt)
		out[5] = clamp(float32(mins) / maxMinutes)
		out[6] = clamp(float32(req.LastTransaction.KmFromCurrent) / maxKm)
	}

	// dim 7: distance from home
	out[7] = clamp(float32(req.Terminal.KmFromHome) / maxKm)

	// dim 8: transaction count in last 24h
	out[8] = clamp(float32(req.Customer.TxCount24h) / maxTxCount24h)

	// dim 9: online transaction flag
	if req.Terminal.IsOnline {
		out[9] = 1
	} else {
		out[9] = 0
	}

	// dim 10: physical card presence
	if req.Terminal.CardPresent {
		out[10] = 1
	} else {
		out[10] = 0
	}

	// dim 11: unknown merchant (1=unknown, 0=known)
	out[11] = 1
	for _, m := range req.Customer.KnownMerchants {
		if m == req.Merchant.ID {
			out[11] = 0
			break
		}
	}

	// dim 12: MCC category risk score (default 0.5 for unknown MCCs)
	if risk, ok := mccRisk[req.Merchant.MCC]; ok {
		out[12] = risk
	} else {
		out[12] = 0.5
	}

	// dim 13: merchant average ticket, normalized
	out[13] = clamp(float32(req.Merchant.AvgAmount) / maxMerchantAvgAmount)
}

// Fast ISO 8601 UTC parsing — zero allocs, no time.Parse overhead
// Expected format: "2026-03-11T18:45:53Z"  (len >= 19)
func atoi2(s string, i int) int {
	return int(s[i]-'0')*10 + int(s[i+1]-'0')
}

func atoi4(s string, i int) int {
	return int(s[i]-'0')*1000 + int(s[i+1]-'0')*100 + int(s[i+2]-'0')*10 + int(s[i+3]-'0')
}

func parseHour(ts string) int {
	if len(ts) < 13 {
		return 0
	}
	return atoi2(ts, 11)
}

// parseDOW returns day-of-week using Tomohiko Sakamoto's algorithm.
// Output: Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
func parseDOW(ts string) int {
	if len(ts) < 10 {
		return 0
	}
	y := atoi4(ts, 0)
	m := atoi2(ts, 5)
	d := atoi2(ts, 8)
	t := [12]int{0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4}
	if m < 3 {
		y--
	}
	// dow: 0=Sun … 6=Sat → remap to Mon=0 … Sun=6
	dow := (y + y/4 - y/100 + y/400 + t[m-1] + d) % 7
	return (dow + 6) % 7
}

// isoToUnix converts "YYYY-MM-DDThh:mm:ssZ" to Unix seconds.
// Uses Howard Hinnant's civil_from_days algorithm — no heap allocs.
func isoToUnix(ts string) int64 {
	if len(ts) < 19 {
		return 0
	}
	y := int64(atoi4(ts, 0))
	mo := int64(atoi2(ts, 5))
	d := int64(atoi2(ts, 8))
	hh := int64(atoi2(ts, 11))
	mm := int64(atoi2(ts, 14))
	ss := int64(atoi2(ts, 17))

	if mo <= 2 {
		y--
		mo += 12
	}
	era := y / 400
	yoe := y - era*400
	doy := (153*mo-457)/5 + d - 1
	doe := yoe*365 + yoe/4 - yoe/100 + doy
	days := era*146097 + doe - 719468

	return days*86400 + hh*3600 + mm*60 + ss
}

func minutesBetween(from, to string) float64 {
	diff := isoToUnix(to) - isoToUnix(from)
	if diff < 0 {
		diff = -diff
	}
	return float64(diff) / 60.0
}

// JSON types
type LastTransaction struct {
	Timestamp     string  `json:"timestamp"`
	KmFromCurrent float64 `json:"km_from_current"`
}

type TxRequest struct {
	ID          string `json:"id"`
	Transaction struct {
		Amount       float64 `json:"amount"`
		Installments int     `json:"installments"`
		RequestedAt  string  `json:"requested_at"`
	} `json:"transaction"`
	Customer struct {
		AvgAmount      float64  `json:"avg_amount"`
		TxCount24h     int      `json:"tx_count_24h"`
		KnownMerchants []string `json:"known_merchants"`
	} `json:"customer"`
	Merchant struct {
		ID        string  `json:"id"`
		MCC       string  `json:"mcc"`
		AvgAmount float64 `json:"avg_amount"`
	} `json:"merchant"`
	Terminal struct {
		IsOnline    bool    `json:"is_online"`
		CardPresent bool    `json:"card_present"`
		KmFromHome  float64 `json:"km_from_home"`
	} `json:"terminal"`
	LastTransaction *LastTransaction `json:"last_transaction"`
}

// HTTP handlers
func handleFraudScore(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	var req TxRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"bad request"}`, http.StatusBadRequest)
		return
	}

	var vec [dims]float32
	vectorize(&req, &vec)

	fraudCount := knn(&vec)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(httpResp[fraudCount])
}

func handleReady(w http.ResponseWriter, r *http.Request) {
	select {
	case <-readyCh:
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	default:
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"status":"loading"}`))
	}
}

// Dataset loading
type refEntry struct {
	Vector [dims]float64 `json:"vector"`
	Label  string        `json:"label"`
}

func loadDataset(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	type reader interface{ Read([]byte) (int, error) }
	var r reader = f

	if len(path) >= 3 && path[len(path)-3:] == ".gz" {
		gz, err := gzip.NewReader(f)
		if err != nil {
			return err
		}
		defer gz.Close()
		r = gz
	}

	var entries []refEntry
	if err := json.NewDecoder(r).Decode(&entries); err != nil {
		return err
	}

	n := len(entries)
	vecs := make([]float32, n*dims)
	fraud := make([]uint8, n)

	for i, e := range entries {
		off := i * dims
		for j := 0; j < dims; j++ {
			vecs[off+j] = float32(e.Vector[j])
		}
		if e.Label == "fraud" {
			fraud[i] = 1
		}
	}

	// Pre-touch all pages → no page-fault latency on first requests
	var chk float32
	for _, v := range vecs {
		chk += v
	}
	log.Printf("Loaded %d reference vectors (checksum=%.4f)", n, chk)

	refVecs = vecs
	refFraud = fraud
	refN = n
	return nil
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "health" {
    os.Exit(0)
	}

	buildResponses()

	log.Printf("Runtime: Go %s | GOMAXPROCS=%d", runtime.Version(), runtime.GOMAXPROCS(0))

	dataPath := os.Getenv("REFERENCES_PATH")
	if dataPath == "" {
		dataPath = "/app/resources/references.json.gz"
	}

	if err := loadDataset(dataPath); err != nil {
		log.Fatalf("Failed to load dataset: %v", err)
	}

	// Signal /ready
	readyOnce.Do(func() { close(readyCh) })

	port := os.Getenv("PORT")
	if port == "" {
		port = "9999"
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/ready", handleReady)
	mux.HandleFunc("/fraud-score", handleFraudScore)

	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           mux,
		ReadTimeout:       5 * time.Second,
		WriteTimeout:      5 * time.Second,
		IdleTimeout:       120 * time.Second,
		ReadHeaderTimeout: 1 * time.Second,
	}

	log.Printf("Listening on :%s — ready", port)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}
