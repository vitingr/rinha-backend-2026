package main

import "time"

const (
	inv23 = 1.0 / 23.0
	inv6  = 1.0 / 6.0
)

// Vectorize transforms a FraudRequest into a 14-dimensional feature vector.
// The vector follows the normalization rules defined in context.md.
func Vectorize(req *FraudRequest, norm *Normalization, mccRisk map[string]float64) [14]float32 {
	var v [14]float32

	// Parse the transaction time once
	t, _ := time.Parse(time.RFC3339, req.Transaction.RequestedAt)

	// 0: amount
	v[0] = float32(clamp(req.Transaction.Amount * InvMaxAmount))

	// 1: installments
	v[1] = float32(clamp(float64(req.Transaction.Installments) * InvMaxInstallments))

	// 2: amount_vs_avg
	v[2] = float32(clamp((req.Transaction.Amount / req.Customer.AvgAmount) * InvAmountVsAvgRatio))

	// 3: hour_of_day (0-23 UTC, normalized by /23)
	v[3] = float32(float64(t.Hour()) * inv23)

	// 4: day_of_week (mon=0, sun=6, normalized by /6)
	// Go's time.Weekday(): Sunday=0, Monday=1, ..., Saturday=6
	// Challenge wants: Monday=0, Sunday=6
	dow := int(t.Weekday())
	if dow == 0 {
		dow = 6 // Sunday → 6
	} else {
		dow-- // Monday=1→0, Tuesday=2→1, etc.
	}
	v[4] = float32(float64(dow) * inv6)

	// 5: minutes_since_last_tx
	if req.LastTx == nil {
		v[5] = -1.0
	} else {
		lt, _ := time.Parse(time.RFC3339, req.LastTx.Timestamp)
		minutes := t.Sub(lt).Minutes()
		v[5] = float32(clamp(minutes * InvMaxMinutes))
	}

	// 6: km_from_last_tx
	if req.LastTx == nil {
		v[6] = -1.0
	} else {
		v[6] = float32(clamp(req.LastTx.KmFromCurrent * InvMaxKm))
	}

	// 7: km_from_home
	v[7] = float32(clamp(req.Terminal.KmFromHome * InvMaxKm))

	// 8: tx_count_24h
	v[8] = float32(clamp(float64(req.Customer.TxCount24h) * InvMaxTxCount24h))

	// 9: is_online
	if req.Terminal.IsOnline {
		v[9] = 1.0
	}

	// 10: card_present
	if req.Terminal.CardPresent {
		v[10] = 1.0
	}

	// 11: unknown_merchant (1 if merchant NOT in known_merchants)
	v[11] = 1.0
	for _, m := range req.Customer.KnownMerchants {
		if m == req.Merchant.ID {
			v[11] = 0.0
			break
		}
	}

	// 12: mcc_risk
	if risk, ok := mccRisk[req.Merchant.MCC]; ok {
		v[12] = float32(risk)
	} else {
		v[12] = 0.5
	}

	// 13: merchant_avg_amount
	v[13] = float32(clamp(req.Merchant.AvgAmount * InvMaxMerchantAvgAmount))

	return v
}

// clamp restricts a value to [0.0, 1.0].
func clamp(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
