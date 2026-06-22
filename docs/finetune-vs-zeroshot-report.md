# Fine-Tuned vs Zero-Shot Voice Cloning — Comparison Report

*Single-speaker English clone of "Daniil" · 6 conditions · N=6 held-out A/B sentences · 2026-06-01*

---

## 1. TL;DR

- **F5-TTS wins both modes and owns the top two slots.** Fine-tuned scores **0.8565**, zero-shot **0.8512** — but the +0.0053 gap is within N=6 noise, so treat them as a **co-leading tie**, not a ranked 1st-vs-2nd.
- **Fine-tuning is engine-dependent.** It barely moved F5 (+0.62% rel score) but gave XTTS the only large lift in the study: **+16.5% rel similarity but only +3.2% rel score (+0.0246 absolute)** — yet XTTS-ft *still* loses to F5 *zero-shot*.
- **The killer cross-engine fact:** F5 with **zero** fine-tuning beats XTTS *with* fine-tuning by **+0.0578 score / +0.1014 similarity (+16% rel)**. Picking the right engine matters far more than fine-tuning the weaker one.
- **This is a similarity contest in disguise.** Naturalness carries the heaviest weight (0.50) but barely discriminates (term spread 0.027); similarity (weight 0.30) drives the ranking (term spread 0.111).

> **The single most important number:** **F5-TTS reaches ~90% of the real-vs-real ceiling** (0.7466 / 0.8284 = 90.1%) — the highest similarity-to-reference of any condition. *(Two caveats up front: (1) the ceiling is an optimistic single same-session pair, so read 90% as a relative cross-engine position, not "90% indistinguishable from a human"; (2) the held-out clips are same-session halves of the same take as training, so this measures **in-session fidelity**, not generalization to a new day / mic / room.)*

---

## 2. What Was Compared & How

**The 6 conditions** — scored on the *same* 6 held-out English A/B sentences (N=6/condition):

| Mode | Engines |
|---|---|
| **Fine-tuned** (both modes exist) | F5-TTS, XTTS-v2 |
| **Zero-shot only** | F5-TTS, XTTS-v2, Chatterbox, GPT-SoVITS v2Pro |

Only **F5 and XTTS** have both modes, so they are the two engines where a true fine-tune-vs-zero-shot delta exists. Chatterbox was held zero-shot by project decision; GPT-SoVITS was demoted to comparison-only (its train config exists but was never run for this eval).

**Only two rows are new.** The four zero-shot rows (and F5/XTTS zero-shot specifically) are the **same generations as the 2026-05-27 N=6 zero-shot A/B**, re-scored here — their numbers match that eval exactly. Only the **two fine-tuned rows** (F5-ft, XTTS-ft) were generated fresh for this comparison. The fight is therefore literally on identical clips for every zero-shot condition.

**Dataset / reference (apples-to-apples).** One long take, sliced to a 216-clip / 33.6-min manifest (184 train / 20 held-out). Every condition — zero-shot *and* fine-tuned — generated from the **identical 9.0 s reference** (`danill_ref9.wav`); fine-tuned rows reuse the exact zero-shot ref + generation params, swapping only the checkpoint, so the delta cleanly isolates fine-tuning. Similarity is measured against a centroid of two 7.05 s held-out real clips. Every clip passes the same front end: mono → −23 LUFS → 16 kHz → peak ≤ −1 dBFS.

**The metric** (`src/voxclone/eval/picker.py` + `config.py`):

```
combined_score = 0.30·sim_eff + 0.50·(UTMOS/5) + 0.20·(1 − WER)
sim_eff = min(similarity / ceiling, 1.0),   ceiling = 0.8284
```

| Term | Weight | Source |
|---|---|---|
| Similarity | 0.30 | ECAPA-TDNN cosine to the centroid of 2 real held-out clips |
| Naturalness | 0.50 | SpeechMOS UTMOS22 (stand-in for sarulab UTMOSv2), MOS 1–5 |
| WER | 0.20 | WhisperX large-v3, Whisper-normalized; disqualify threshold 0.20 |

The **ceiling 0.8284** is the mean pairwise cosine of the real held-out clips — but with only 2 clips that is exactly *one* same-session pair, so it is optimistic. The cross-engine **ranking** is fair (shared reference + front end); the absolute **%ceiling** is biased high. The formula reproduces all six stored scores to machine precision.

---

## 3. Headline Results

| rank | engine | mode | similarity | % ceiling | UTMOS | WER | score |
|---|---|---|---|---|---|---|---|
| 1 | F5-TTS | fine-tuned | 0.7466 | 90.1% | 3.878 | 0.008 | **0.8565** |
| 2 | F5-TTS | zero-shot | 0.7356 | 88.8% | 3.865 | 0.008 | **0.8512** |
| 3 | XTTS-v2 | fine-tuned | 0.6342 | 76.6% | 3.637 | 0.000 | 0.7934 |
| 4 | XTTS-v2 | zero-shot | 0.5445 | 65.7% | 3.716 | 0.000 | 0.7688 |
| 5 | Chatterbox | zero-shot | 0.5557 | 67.1% | 3.610 | 0.008 | 0.7606 |
| 6 | GPT-SoVITS | zero-shot | 0.4403 | 53.2% | 3.630 | 0.008 | 0.7207 |

The two F5 rows sit **~0.06–0.10 above** the rest of the field; the four non-F5 rows cluster within a 0.073 band. **What survives N=6:** (a) both F5 rows lead the field; (b) XTTS-ft > XTTS-zs (+0.0246, similarity-driven); (c) GPT-SoVITS is clearly last (~0.040 below row 5). **Treated as ties:** F5-ft vs F5-zs (gap 0.0053), and rows 4/5 (XTTS-zs vs Chatterbox, gap 0.0082 — they even invert on raw similarity, 0.5445 < 0.5557).

*Footnote: GPT-SoVITS = 0.72075, shown as 0.7207.*

---

## 4. Fine-Tune vs Zero-Shot Deltas

| engine | metric | zero-shot | fine-tuned | abs Δ | % rel Δ |
|---|---|---|---|---|---|
| F5-TTS | similarity | 0.7356 | 0.7466 | +0.0110 | +1.5% |
| F5-TTS | % ceiling | 88.8% | 90.1% | +1.3 pt | — |
| F5-TTS | UTMOS | 3.865 | 3.878 | +0.0132 | +0.3% |
| F5-TTS | WER | 0.008 | 0.008 | 0.000 | 0% |
| F5-TTS | **score** | 0.8512 | 0.8565 | **+0.0053** | **+0.62%** |
| XTTS-v2 | similarity | 0.5445 | 0.6342 | +0.0897 | +16.5% |
| XTTS-v2 | % ceiling | 65.7% | 76.6% | +10.8 pt | — |
| XTTS-v2 | UTMOS | 3.716 | 3.637 | −0.0788 | −2.1% |
| XTTS-v2 | WER | 0.000 | 0.000 | 0.000 | 0% |
| XTTS-v2 | **score** | 0.7688 | 0.7934 | **+0.0246** | **+3.2%** |

**F5 — fine-tuning bought a tie.** F5 was already at 88.8% of ceiling zero-shot, leaving almost no headroom. Every sub-metric moved a negligible amount; WER was identical. **The +0.0053 score gap is within N=6 noise — do not claim fine-tuning helps F5.** No confidence intervals were computed for any row; only per-clip UTMOS std exists, and this delta is far below the logged per-condition UTMOS std (0.096–0.219).

**XTTS — a real lift, but a trade.** Fine-tuning produced the biggest similarity gain in the study (+0.0897, +16.5% rel), the lever an autoregressive GPT responds to. But UTMOS *dropped* ~2% — the only condition where fine-tuning hurt naturalness. Net score still rose because under the weights the similarity gain (0.30·0.0897 = +0.0269) outweighs the naturalness loss (0.50·(−0.0788/5) = −0.0079). The lift was real (+0.0246 score, ~4.6× F5's), but XTTS-ft never left **3rd place**.

**Cross-engine fact (the headline).** F5 *zero-shot* out-scores the best XTTS checkpoint on every axis that matters: similarity 0.7356 vs 0.6342 (**+0.1014, +16% rel**), score 0.8512 vs 0.7934 (**+0.0578**). Notably, F5-zs's similarity *lead* over XTTS-ft (+0.1014) is **larger than the entire benefit XTTS got from fine-tuning** (+0.0897). For this speaker, **engine choice beats fine-tuning the weaker engine** (switching XTTS→F5 = +0.0578 vs fine-tuning XTTS = +0.0246).

---

## 5. Metric-by-Metric

| score term | weight | spread across 6 conditions | range | discriminating? |
|---|---|---|---|---|
| Similarity (0.30·sim_eff) | 0.30 | **0.111** | 0.1595 – 0.2704 | **YES — drives the ranking** |
| Naturalness (0.50·UTMOS/5) | 0.50 | 0.027 | 0.3610 – 0.3878 | no — saturated |
| WER (0.20·(1−WER)) | 0.20 | 0.0017 | 0.1983 – 0.2000 | no — pass/fail floor |

**Similarity** is the only real discriminator. The similarity-term spread (0.111) is **4.1× the naturalness-term spread** and **~66× the WER spread** — even though naturalness carries 1.67× the weight.

**Naturalness** is saturated. All six engines land **3.61–3.88**, and *every one out-scores the real Daniil's own held-out UTMOS of 3.295.* That every synthetic clip is rated more "natural" than the genuine human is a known auto-MOS saturation/synthetic-bias artifact, partly attributable to using UTMOS22 as a stand-in for the intended UTMOSv2. The 0.50-weighted term barely moves the needle.

**WER** is a near-constant floor (all 0.000–0.008, nobody near the 0.20 disqualify threshold) — a pass/fail check, not a ranker.

**The decomposition insight:** despite naturalness carrying the heaviest weight, the metric is *in practice a similarity ranking with a naturalness tie-break and a WER floor.* The engine whose architecture best moves speaker similarity wins.

---

## 6. Why the Results Look This Way

**F5-TTS — saturated zero-shot, no headroom.** Non-autoregressive flow-matching does timbre transfer *in-context* from the reference clip at inference, so identity is copied rather than stored in trainable weights. Zero-shot already lands at 88.8% of ceiling; fine-tuning had almost nothing left to adapt (+0.0110 similarity). The tiny delta reflects the architecture front-loading the work into inference-time conditioning — not a fine-tuning failure.

**XTTS-v2 — similarity lift + naturalness dip.** A GPT-style autoregressive token LM carries speaker identity through conditioning latents. Fine-tuning re-points that LM at the speaker — exactly the lever similarity responds to, hence the +16.5% rel lift. But an AR LM also learns the speaker's prosody/pacing, and on a small split it plausibly over-fits, which is the most likely reason naturalness dipped (−0.079 UTMOS). *Provenance caveat: the loss/step figures sometimes cited for this run — "train loss 0.94 vs eval loss 3.28", an alternative "train ~2.8 vs eval ~3.26" mel-CE framing, and "best checkpoint near step 936" — all come from uncommitted training notes and carry equal (un)verified weight; none appear in committed artifacts. Treat the over-fit as suggestive, not proven; XTTS-ft still scored above XTTS-zs, so "over-fit" did not mean a worse clone. Some of the −0.079 is also run-to-run variance, since XTTS is sampled (temp 0.7, unseeded) while F5 is deterministic (seed 42).*

**GPT-SoVITS — structural English ceiling.** Chinese-origin v2Pro. For English, the BERT context features that drive prosody are zero-padded (flatter prosody, weak question-vs-statement intonation) and the speech tokenizer is a Chinese-trained HuBERT (mismatched unit inventory). These are baked into the front end and tokenizer, *not* the speaker-specific weights fine-tuning would touch — a structural ceiling the documentation reports fine-tuning "cannot remove." Hence last place (0.4403 similarity, 53.2% ceiling) and the highest output variance (UTMOS std 0.219). *Its fine-tune was never run, so this is the documented rationale for not trying, not an empirical wall.*

**Chatterbox — mid-pack, zero-shot by rule.** A 0.5B Llama-backbone zero-shot TTS, never fine-tuned (project decision). It ties XTTS-zs on similarity (67.1% vs 65.7% ceiling) but carries the lowest UTMOS of the six (3.610) and the second-highest variance (std 0.181, behind GPT-SoVITS), consistent with its sampled, unseeded generation. Net rank 5, just under XTTS-zs.

---

## 7. Cost / ROI / When to Fine-Tune

**Cost ledger.** Zero-shot = one 9.0 s reference clip + one inference call. Fine-tune = a 33.6-min single-take capture → 216-clip manifest → 184 train clips → a *per-engine* GPU run (RTX 5090 24 GB; XTTS 15 epochs / eff-batch 6 / lr 5e-6 / fp32; F5 100 epochs / lr 1e-5 / 3200-frame batches / 300 warmup) → plus overfit-management overhead (best-checkpoint selection for XTTS). *The lr / epochs / batch settings come from the committed train configs; any step counts (e.g. "~1150 steps") are from uncommitted training notes, not artifacts.*

| Decision | Capture | GPU run | Overfit risk | Output rank | Verdict |
|---|---|---|---|---|---|
| **F5 zero-shot (SHIP)** | 9.0 s ref | none | none | 1st-tie (0.8512) | All the quality, none of the cost |
| F5 fine-tuned | 33.6 min | 100 epochs | low | 1st-tie (0.8565) | +0.0053 = a tie; a near-free bonus at best |
| XTTS fine-tuned | 33.6 min | 15 epochs (lr 5e-6) | higher (UTMOS −0.079) | 3rd (0.7934) | Real +16.5% sim lift, still loses to F5-zs |
| Fine-tune GPT-SoVITS | 33.6 min | won't help | structural wall | last (0.7207) | Don't — structural English ceiling |

**Was it worth it?** On the winning engine, **no** — fine-tuning F5 produced gains too small to be decision-relevant at this N. On XTTS, the lift was genuine but moved a 3rd-place engine up *within* 3rd place. The metric also structurally *under*-prices fine-tuning's documented long-form wins (identity consistency across varied text, long-text stability, learned pacing/breath) — a 6-sentence test cannot see them.

**Decision rule:**
- **Ship zero-shot** when the best engine already sits near the ceiling zero-shot (F5 ~89%) and the deliverable is short/varied clips. *(This project.)*
- **Fine-tune** only when (a) the best available engine is a GPT-AR / conditioning-latent type that starts far from ceiling **and** you have 15–30 min of clean data, or (b) the deliverable is long-form/identity-consistency narration the short-clip metric is blind to.
- **Never** fine-tune to fix structure (GPT-SoVITS English ceiling) or where project policy forbids it (Chatterbox). Returns diminish past ~15–30 min; overfit risk climbs beyond.

---

## 8. Caveats & Validity

| Caveat | Affects RANKING or ABSOLUTE-only? |
|---|---|
| **N=6, no confidence intervals** (only per-clip UTMOS std 0.096–0.219). Score gaps under ~0.02 are ties: F5-ft vs F5-zs (+0.0053) and XTTS-zs vs Chatterbox (+0.0082). | Ranking confidence at margins < ~0.02 |
| **Ceiling is one same-session pair** (2 clips = 1 pair, two 7.05 s halves of a 14.1 s take), so absolute %ceiling is optimistic — "90% of ceiling" ≠ "90% indistinguishable from human." | Absolute only (monotonic divisor; ranking fair) |
| **Same-session eval** — held-out clips are from the *same recording session* as training (halves of the same 14.1 s take); this measures in-session fidelity, **not** generalization to a new day / mic / room. The ship recommendation rests on a similarity metric that is itself same-session-biased. | Absolute generalization claims |
| **Similarity is reference-biased** — ECAPA cosine to a fixed same-speaker centroid; rewards matching *this* recording, not validated listener identity. Shared ref makes it symmetric. | Absolute only |
| **UTMOS is a saturated stand-in** (UTMOS22, not UTMOSv2); all 6 engines beat the real speaker's 3.295. Absolute MOS untrustworthy. | Absolute strongly; ranking at tie margins |
| **Sampling asymmetry** — F5 deterministic (seed 42); XTTS/Chatterbox/GPT-SoVITS single unseeded draws (temp 0.7–0.8) that would shift on re-run. | Ranking, for the sampled engines |
| **No human listening test** — entire verdict rests on automatic proxies (ECAPA + UTMOS + WhisperX), unvalidated against listener preference for this speaker. A single quick human A/B (a few minutes of paired listening on the 6 sentences) would settle the F5-ft-vs-zs tie and the saturated-naturalness question; it is **not yet planned**. | Ranking-vs-perception confidence |
| **Unverified training-note figures** — XTTS "train 0.94 / eval 3.28" (and the "2.8 / 3.26" mel-CE variant), "157/29 rows", "step 936", "~1150 steps", and F5 "185 rows" are NOT in committed artifacts; cite as training-run notes, not audited metrics. | Neither (provenance) |

**Per-sentence win-rate (the missing CI substitute).** At N=6 the strongest available stand-in for a confidence interval is a sentence-level win count — *did F5-ft beat F5-zs on 6/6 or on 3/3 of the six sentences?* The committed artifacts store only per-condition means and per-clip UTMOS std, **not** the per-sentence similarity/score arrays, so this breakdown is **not recoverable** from the current data pack. It would be the single highest-value addition to harden (or properly kill) the F5-ft "tie" claim and should be captured in the next eval run.

**Ranking stability across evals.** The F5 #1 result is **consistent across all three evals to date** — the N=1 single-sentence zero-shot eval (F5 0.8405, top), the N=6 zero-shot A/B (F5 0.8512, top), and this N=6 fine-tune comparison (F5 #1 in both modes). F5 has ranked first every time; this cross-eval consistency is the closest thing to repeatability the study currently has (though all three share the same speaker, session, and metric, so they are not independent replications).

**What survives every caveat (the defensible core):** (1) **F5 leads both modes**, and **F5 zero-shot beats XTTS fine-tuned** by 0.1014 similarity / 0.0578 score — an order of magnitude larger than any noise margin; (2) **fine-tuning helps XTTS a lot** (+16.5% rel similarity) but **barely moves F5** (near-zero headroom). Everything else, including any F5 fine-tune "win," is within noise. *(These three facts are the canonical claim referenced throughout — for a slide deck, show them once.)*

---

## 9. Presentation Kit

**Recommended charts (3, each proving one claim):**

1. **Ranking bar (Chart 1).** Horizontal bars, sorted by score, **all 6 rows**. X-axis = composite score **0.70–0.90** (*footnote the truncated axis on-slide*); color fine-tuned vs zero-shot so the two F5 bars sit together at top. *Proves: a clear ranking exists; F5 owns the top two slots.*
2. **% of ceiling (Chart 2 — the money chart).** Horizontal bars, X-axis = **0–100% of ceiling**, dashed reference line at 100% labeled "real Daniil (ceiling)", **all 6 rows**. *Proves: similarity is what separates the engines; F5 sits near the human line, GPT-SoVITS barely past half.* (Note: rows reorder vs Chart 1 — Chatterbox 67.1% edges XTTS-zs 65.7% on ceiling but loses on score; a talking point in itself.)
3. **Fine-tune slope (Chart 3).** Two-point slope (zero-shot → fine-tuned) on the %-ceiling axis, **F5 and XTTS only**. XTTS climbs steeply (+10.8 pt); F5 is nearly flat at the top (+1.3 pt). Callout: "F5 zero-shot (88.8%) beats XTTS fine-tuned (76.6%)." *Proves: fine-tune payoff depends entirely on the engine.*

**Narrative arc:** Verdict → Method (one fair fight, identical zero-shot clips) → Chart 1 (ranking) → Chart 2 (similarity is the splitter) → Chart 3 (the counterintuitive lift slopes) → Why (architecture) → honest caveats → Ship recommendation.

**Ready-to-paste speaker talking points:**
- "F5-TTS clones Daniil at ~90% of our ceiling — and it's the top engine whether or not we train it. Its trained and untrained versions are a statistical tie at the top."
- "Same 6 sentences, same 9-second reference, same audio pipeline for every engine. The four zero-shot rows are literally the clips from our earlier eval, re-scored — only the two fine-tuned versions are new. One score: 30% similarity + 50% naturalness + 20% intelligibility. A fair fight."
- "Here's the surprise: training F5 barely moved it (+1.3 pts) — it was already near the ceiling. Training XTTS moved it a lot (+11 pts), and it *still* lost to F5 with no training at all."
- "F5 transfers a voice from context, so it starts near the top. XTTS learns the speaker by training, so it has room to grow. GPT-SoVITS has a built-in English handicap no training removes."
- "F5 has come first in every eval we've run — one sentence, six sentences, and now fine-tuned vs zero-shot. The ranking is stable."
- "Six sentences, not six hundred — directional, not final. The ceiling is optimistic, our naturalness meter is saturated, and we've only tested same-session clips. We're showing you the warts on purpose."
- "Ship F5-TTS zero-shot: 99.4% of the fine-tuned score, no training run, one 9-second clip. Keep the fine-tune only as a near-free bonus. Don't chase XTTS or GPT-SoVITS."

---

## 10. Bottom Line

**Ship F5-TTS in zero-shot mode.** It scores **0.8512 — 99.4% of the fine-tuned checkpoint** and ~90% of the (optimistic) real-vs-real ceiling — from a single 9-second reference and one inference call, with no training, no overfit risk, and deterministic output. **Keep the fine-tuned F5 checkpoint only as a near-free, statistically-tied bonus** if a pipeline already exists; do not claim it as a real gain. **Do not invest training effort** in XTTS (best lift in the study, but still below F5 zero-shot) or GPT-SoVITS (structural English ceiling fine-tuning cannot fix). The decisive lesson for this speaker: **choosing the right engine beats fine-tuning the wrong one.** *(Directional, not final: N=6, no human listening test, same-session clips only.)*
