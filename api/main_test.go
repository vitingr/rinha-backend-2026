package main

import (
	"bytes"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

func almostEq(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) <= tol
}

// ─────────────────────────────────────────────────────────────────────────────
// Precomputed responses
// ─────────────────────────────────────────────────────────────────────────────

func TestBuildResponses(t *testing.T) {
	buildResponses()
	cases := []struct {
		n    int
		want string
	}{
		{0, `{"approved":true,"fraud_score":0.0}`},
		{1, `{"approved":true,"fraud_score":0.2}`},
		{2, `{"approved":true,"fraud_score":0.4}`},
		{3, `{"approved":false,"fraud_score":0.6}`},
		{4, `{"approved":false,"fraud_score":0.8}`},
		{5, `{"approved":false,"fraud_score":1.0}`},
	}
	for _, c := range cases {
		got := string(httpResp[c.n])
		if got != c.want {
			t.Errorf("httpResp[%d] = %q, want %q", c.n, got, c.want)
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Clamp
// ─────────────────────────────────────────────────────────────────────────────

func TestClamp(t *testing.T) {
	tests := []struct{ in, want float32 }{
		{-99, 0}, {0, 0}, {0.5, 0.5}, {1, 1}, {1.5, 1}, {99, 1},
	}
	for _, c := range tests {
		if got := clamp(c.in); got != c.want {
			t.Errorf("clamp(%v)=%v want %v", c.in, got, c.want)
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Date/time helpers
// ─────────────────────────────────────────────────────────────────────────────

func TestParseHour(t *testing.T) {
	cases := []struct {
		ts   string
		want int
	}{
		{"2026-03-11T18:45:53Z", 18},
		{"2026-03-14T05:15:12Z", 5},
		{"2026-01-01T00:00:00Z", 0},
		{"2026-12-31T23:59:59Z", 23},
	}
	for _, c := range cases {
		if got := parseHour(c.ts); got != c.want {
			t.Errorf("parseHour(%q)=%d want %d", c.ts, got, c.want)
		}
	}
}

func TestParseDOW(t *testing.T) {
	cases := []struct {
		ts   string
		want int
	}{
		{"2026-03-09T00:00:00Z", 0}, // Mon
		{"2026-03-10T00:00:00Z", 1}, // Tue
		{"2026-03-11T00:00:00Z", 2}, // Wed
		{"2026-03-12T00:00:00Z", 3}, // Thu
		{"2026-03-13T00:00:00Z", 4}, // Fri
		{"2026-03-14T00:00:00Z", 5}, // Sat
		{"2026-03-15T00:00:00Z", 6}, // Sun
	}
	for _, c := range cases {
		if got := parseDOW(c.ts); got != c.want {
			t.Errorf("parseDOW(%q)=%d want %d", c.ts, got, c.want)
		}
	}
}

func TestIsoToUnix(t *testing.T) {
  if v := isoToUnix("1970-01-01T00:00:00Z"); v != 0 {
    t.Errorf("epoch = %d, want 0", v)
  }
  if v := isoToUnix("1970-01-01T00:01:00Z"); v != 60 {
    t.Errorf("1 min = %d, want 60", v)
  }
  // time.Date(2026,3,11,18,45,53,0,time.UTC).Unix() = 1773254753
  want := int64(1773254753) 
  if got := isoToUnix("2026-03-11T18:45:53Z"); got != want {
    t.Errorf("isoToUnix = %d, want %d", got, want)
  }
}

func TestMinutesBetween(t *testing.T) {
	from := "2026-03-11T14:58:35Z"
	to := "2026-03-11T18:45:53Z"
	// |18:45:53 - 14:58:35| = 3h47m18s = 227.3 min
	want := (3*3600.0 + 47*60.0 + 18.0) / 60.0
	got := minutesBetween(from, to)
	if math.Abs(got-want) > 0.001 {
		t.Errorf("minutesBetween = %.4f, want %.4f", got, want)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Vectorization — ground truth from DETECTION_RULES.md
// ─────────────────────────────────────────────────────────────────────────────

func TestVectorizeLegitTx(t *testing.T) {
	// Expected: [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
	var got [dims]float32
	vectorize(makeLegitReq(), &got)
	want := [dims]float32{0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006}
	for i, w := range want {
		if !almostEq(got[i], w, 0.002) {
			t.Errorf("dim[%d]: got %.5f want %.5f", i, got[i], w)
		}
	}
}

func TestVectorizeFraudTx(t *testing.T) {
	// Expected: [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]
	var got [dims]float32
	vectorize(makeFraudReq(), &got)
	want := [dims]float32{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}
	for i, w := range want {
		if !almostEq(got[i], w, 0.002) {
			t.Errorf("dim[%d]: got %.5f want %.5f", i, got[i], w)
		}
	}
}

func TestVectorizeWithLastTx(t *testing.T) {
	req := makeLegitReq()
	req.LastTransaction = &LastTransaction{
		Timestamp:     "2026-03-11T14:58:35Z",
		KmFromCurrent: 18.8626479774,
	}
	var got [dims]float32
	vectorize(req, &got)
	if got[5] == -1 {
		t.Error("dim[5] should not be -1 when LastTransaction is set")
	}
	if got[6] == -1 {
		t.Error("dim[6] should not be -1 when LastTransaction is set")
	}
	// dim6: 18.8626/1000 ≈ 0.0189
	if !almostEq(got[6], 0.0189, 0.001) {
		t.Errorf("dim[6]=%.5f want ~0.0189", got[6])
	}
}

func TestVectorizeUnknownMCC(t *testing.T) {
	req := makeLegitReq()
	req.Merchant.MCC = "9999"
	var got [dims]float32
	vectorize(req, &got)
	if got[12] != 0.5 {
		t.Errorf("unknown MCC dim[12]=%.2f want 0.5", got[12])
	}
}

func TestVectorizeClampOverflow(t *testing.T) {
	req := makeLegitReq()
	req.Transaction.Amount = 99999
	req.Terminal.KmFromHome = 99999
	var got [dims]float32
	vectorize(req, &got)
	if got[0] != 1.0 {
		t.Errorf("dim[0] (amount) should clamp to 1.0, got %.4f", got[0])
	}
	if got[7] != 1.0 {
		t.Errorf("dim[7] (km_from_home) should clamp to 1.0, got %.4f", got[7])
	}
}

func TestVectorizeOnlineNoCard(t *testing.T) {
	req := makeLegitReq()
	req.Terminal.IsOnline = true
	req.Terminal.CardPresent = false
	var got [dims]float32
	vectorize(req, &got)
	if got[9] != 1 {
		t.Error("dim[9] (is_online) should be 1")
	}
	if got[10] != 0 {
		t.Error("dim[10] (card_present) should be 0")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// KNN
// ─────────────────────────────────────────────────────────────────────────────

func TestKNNAllLegit(t *testing.T) {
	setupDataset(20, false)
	var q [dims]float32
	if n := knn(&q); n != 0 {
		t.Errorf("all-legit: got %d fraud neighbors, want 0", n)
	}
}

func TestKNNAllFraud(t *testing.T) {
	setupDataset(20, true)
	var q [dims]float32
	if n := knn(&q); n != kNeighbors {
		t.Errorf("all-fraud: got %d fraud neighbors, want %d", n, kNeighbors)
	}
}

func TestKNNMixed(t *testing.T) {
	// 3 fraud at origin, 10 legit at dim0=1.0
	// Nearest 5: {0,1,2}=fraud (dist²=0) + {3,4}=legit (dist²=1)
	n := 13
	refVecs = make([]float32, n*dims)
	refFraud = make([]uint8, n)
	refN = n
	refFraud[0], refFraud[1], refFraud[2] = 1, 1, 1
	for i := 3; i < n; i++ {
		refVecs[i*dims] = 1.0
	}
	var q [dims]float32
	if fc := knn(&q); fc != 3 {
		t.Errorf("mixed: fraud count = %d, want 3", fc)
	}
}

func TestKNNFewerThanK(t *testing.T) {
	// Dataset with only 3 entries — heap.size stays < k
	setupDataset(3, true)
	var q [dims]float32
	if n := knn(&q); n != 3 {
		t.Errorf("tiny dataset: fraud count = %d, want 3", n)
	}
}

func TestEuclidSq(t *testing.T) {
	vecs := make([]float32, dims)
	var q [dims]float32
	if d := euclidSq(&q, vecs, 0); d != 0 {
		t.Errorf("self dist = %v, want 0", d)
	}
	vecs[0] = 1.0
	if d := euclidSq(&q, vecs, 0); d != 1.0 {
		t.Errorf("unit dist² = %v, want 1.0", d)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP integration
// ─────────────────────────────────────────────────────────────────────────────

func TestHandleReadyOK(t *testing.T) {
	buildResponses()
	setupDataset(10, false)
	readyOnce.Do(func() { close(readyCh) })

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	w := httptest.NewRecorder()
	handleReady(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("ready = %d, want 200", w.Code)
	}
}

func TestHandleFraudScoreOK(t *testing.T) {
	buildResponses()
	setupDataset(20, false)

	body, _ := json.Marshal(makeLegitReq())
	req := httptest.NewRequest(http.MethodPost, "/fraud-score", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handleFraudScore(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON: %v — %s", err, w.Body.String())
	}
	if _, ok := resp["approved"]; !ok {
		t.Error("missing 'approved'")
	}
	if _, ok := resp["fraud_score"]; !ok {
		t.Error("missing 'fraud_score'")
	}
}

func TestHandleFraudScoreAllFraud(t *testing.T) {
	buildResponses()
	setupDataset(20, true)

	body, _ := json.Marshal(makeFraudReq())
	req := httptest.NewRequest(http.MethodPost, "/fraud-score", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handleFraudScore(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d", w.Code)
	}
	if !strings.Contains(w.Body.String(), `"approved":false`) {
		t.Errorf("expected denied in all-fraud, got: %s", w.Body.String())
	}
}

func TestHandleFraudScoreBadJSON(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/fraud-score", strings.NewReader("{bad"))
	w := httptest.NewRecorder()
	handleFraudScore(w, req)
	if w.Code != http.StatusBadRequest {
		t.Errorf("bad JSON = %d, want 400", w.Code)
	}
}

func TestHandleFraudScoreWrongMethod(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/fraud-score", nil)
	w := httptest.NewRecorder()
	handleFraudScore(w, req)
	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("GET = %d, want 405", w.Code)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmarks
// ─────────────────────────────────────────────────────────────────────────────

func BenchmarkVectorize(b *testing.B) {
	req := makeFullReq()
	var out [dims]float32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vectorize(req, &out)
	}
}

func BenchmarkEuclidSq(b *testing.B) {
	vecs := make([]float32, dims*2)
	for i := range vecs {
		vecs[i] = float32(i) * 0.01
	}
	var q [dims]float32
	for i := range q {
		q[i] = 0.5
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = euclidSq(&q, vecs, 0)
	}
}

func BenchmarkKNN100k(b *testing.B) {
	setupDataset100k()
	var q [dims]float32
	for i := range q {
		q[i] = 0.5
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = knn(&q)
	}
}

func BenchmarkFullHTTPRequest(b *testing.B) {
	buildResponses()
	setupDataset100k()
	readyOnce.Do(func() { close(readyCh) })

	body, _ := json.Marshal(makeFullReq())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest(http.MethodPost, "/fraud-score", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		handleFraudScore(w, req)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixtures
// ─────────────────────────────────────────────────────────────────────────────

func makeLegitReq() *TxRequest {
	req := &TxRequest{}
	req.Transaction.Amount = 41.12
	req.Transaction.Installments = 2
	req.Transaction.RequestedAt = "2026-03-11T18:45:53Z"
	req.Customer.AvgAmount = 82.24
	req.Customer.TxCount24h = 3
	req.Customer.KnownMerchants = []string{"MERC-003", "MERC-016"}
	req.Merchant.ID = "MERC-016"
	req.Merchant.MCC = "5411"
	req.Merchant.AvgAmount = 60.25
	req.Terminal.IsOnline = false
	req.Terminal.CardPresent = true
	req.Terminal.KmFromHome = 29.23
	req.LastTransaction = nil
	return req
}

func makeFraudReq() *TxRequest {
	req := &TxRequest{}
	req.Transaction.Amount = 9505.97
	req.Transaction.Installments = 10
	req.Transaction.RequestedAt = "2026-03-14T05:15:12Z"
	req.Customer.AvgAmount = 81.28
	req.Customer.TxCount24h = 20
	req.Customer.KnownMerchants = []string{"MERC-008", "MERC-007", "MERC-005"}
	req.Merchant.ID = "MERC-068"
	req.Merchant.MCC = "7802"
	req.Merchant.AvgAmount = 54.86
	req.Terminal.IsOnline = false
	req.Terminal.CardPresent = true
	req.Terminal.KmFromHome = 952.27
	req.LastTransaction = nil
	return req
}

func makeFullReq() *TxRequest {
	req := makeLegitReq()
	req.LastTransaction = &LastTransaction{
		Timestamp:     "2026-03-11T14:58:35Z",
		KmFromCurrent: 18.8626479774,
	}
	return req
}

func setupDataset(n int, allFraud bool) {
	refVecs = make([]float32, n*dims)
	refFraud = make([]uint8, n)
	refN = n
	if allFraud {
		for i := range refFraud {
			refFraud[i] = 1
		}
	}
}

func setupDataset100k() {
	n := 100_000
	vecs := make([]float32, n*dims)
	fraud := make([]uint8, n)
	for i := 0; i < n; i++ {
		off := i * dims
		for j := 0; j < dims; j++ {
			vecs[off+j] = float32((i*dims+j)%100) / 100.0
		}
		if i%3 == 0 {
			fraud[i] = 1
		}
	}
	refVecs = vecs
	refFraud = fraud
	refN = n
}
