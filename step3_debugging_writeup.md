# Step 3 Debugging Log: County Climate Feature Computation

Four distinct bugs surfaced while building `compute_county_climate.py` — the
step that aggregates 128M daily station observations into county-level
annual climate features (GDD, heat stress days, precipitation) via
inverse-distance-weighted (IDW) interpolation across each county's 5 nearest
weather stations.

They're grouped here because they share a common shape, which is itself
the most useful takeaway: **pandas has several places where "no data"
silently becomes a plausible-looking number instead of `NaN`**, and each
one looks like clean, correct output until you go looking for it.

---

## Bug 1 — Boolean inversion on an `object`-dtype column

**Symptom:**
```
KeyError: "None of [Index([-2, -2, -2, ...], dtype='object', length=463547)] are in the [index]"
```

**Root cause:** After a `how="left"` merge, county-station pairs with no
matching station-year got `NaN` in the `coverage_ok` column. `NaN` can't
coexist with real `bool` values in one dtype, so pandas silently upcast the
whole column to `object` — a column of literal Python `True`/`False`
objects, not real boolean dtype. `.fillna(False)` filled the NaNs correctly
but never restored the dtype. Applying `~` (invert) to an `object`-dtype
column of Python bools doesn't do logical negation — Python's `bool` is a
subclass of `int`, so it falls through to **bitwise NOT on the underlying
int** (`~True == -2`, `~False == -1`). Pandas then tried to interpret those
`-2`s as a list of index labels to select, which don't exist.

**Fix:** force real boolean dtype before inverting:
```python
coverage_ok = merged["coverage_ok"].fillna(False).astype(bool)
poor_coverage = ~coverage_ok
```

---

## Bug 2 — `.sum()` on an all-`NaN` group defaults to `0.0`, not `NaN`

**Symptom:** County-years with **zero** valid stations (`stations_used = 0`)
were reporting `tmax_avg = 0.0°C` instead of null — a fabricated "it was
freezing" reading standing in for "we have no idea."

**Root cause:** `.mean()` correctly returns `NaN` for an all-`NaN` group,
but `.sum()`'s default behavior is different: summing zero (or all-`NaN`)
values returns `0.0`, not `NaN`. When every station backing a county-year
had `weight_norm = 0` (because none passed validity), the weighted sum of
`NaN * 0` still collapsed to a real `0.0` under `.sum()`.

**Fix (first pass):** explicitly null out feature columns wherever
`stations_used == 0` after aggregation. (This fix was later superseded by
Bug 3's more precise version — see below.)

---

## Bug 3 — A shared "valid" flag let partial-data stations fabricate zeros for features they didn't have

**Symptom:** A `sparse` (1-station) county-year showed `tmax_avg = 0.0`
even though its one contributing station had **zero temperature readings**
that year — it only had real precipitation data.

**Root cause:** `merged["valid"] = merged[feature_cols].notna().any(axis=1)`
marked a station "valid" if it had data for **any one** of six features.
A station with real precipitation but no temperature got counted as a full
contributor to `tmax_avg`, `gdd`, and `heat_stress` too — and since its
values there were `NaN`, the `.sum()` default from Bug 2 kicked in again
and fabricated `0.0` for those features specifically.

**Fix:** compute IDW weights and aggregate **per feature, independently** —
a station can be a valid contributor to precipitation without being a
valid contributor to temperature in the same year:
```python
for col in feature_cols:
    valid = merged[col].notna()
    weight_if_valid = merged["weight"].where(valid)
    weight_sum = weight_if_valid.groupby([...]).transform("sum")
    weight_norm = weight_if_valid / weight_sum
    weighted_val = merged[col] * weight_norm
    agg = weighted_val.groupby([...]).sum(min_count=1)  # NaN, never a fabricated 0.0
```
`stations_used` (the overall confidence label) is now the **minimum**
contributor count across all six features — a county-year is only as
trustworthy as its worst-covered feature.

### Bug 3b — the fix's own cleanup step then destroyed good data

**Symptom:** After the Bug 3 fix, `prcp_total` — which *did* have real
data — was showing `NaN` too.

**Root cause:** A leftover blanket step nulled **every** feature whenever
`climate_quality == "missing"`. Since that label is driven by the *worst*
feature (temperature, in this case), it was overwriting genuinely good
data in unrelated columns (precipitation) that had nothing to do with why
the label said "missing."

**Fix:** delete the blanket null-out entirely. Each feature's own
`sum(min_count=1)` already handles its own nulling correctly, column by
column — no shared gate needed. `climate_quality` remains a confidence
*label* for the row, not a gate on any individual column's data.

---

## Bug 4 — `NaN > threshold` evaluates to `False`, not `NaN`

**Symptom:** Even after Bugs 2/3 were fixed, `gdd` and `heat_stress` were
still showing `0.0` instead of `NaN` for a station-year with no TMAX data.

**Root cause:** two compounding issues in `compute_station_features`:
1. `(pivoted["TMAX"] > HEAT_THRESHOLD).astype(float)` — in pandas,
   `NaN > 34` evaluates to `False`, not `NaN`. A day with no temperature
   reading silently became "confirmed not a heat stress day" (`0.0`)
   instead of "unknown."
2. Even where the math *did* correctly propagate `NaN` (e.g. `gdd_day`'s
   arithmetic), the aggregation step `.agg(gdd=("gdd_day", "sum"))` hit
   the same all-`NaN`-group-defaults-to-`0.0` issue from Bug 2 — one
   level upstream of where it was first found and fixed.

**Fix:**
```python
pivoted["heat_stress_day"] = np.where(
    pivoted["TMAX"].isna(), np.nan, (pivoted["TMAX"] > HEAT_THRESHOLD).astype(float)
)
pivoted["prcp_day"] = np.where(
    pivoted["PRCP"].isna(), np.nan, (pivoted["PRCP"] > 0).astype(float)
)
```
plus `sum(min_count=1)` instead of plain `"sum"` in the station-level
aggregation for `prcp_total`, `gdd`, `heat_stress`, and `prcp_days`.

---

## The common thread

Every one of these bugs is the same failure mode wearing a different
disguise: **pandas has multiple, inconsistent defaults for "empty" or
"missing"** — `.mean()` returns `NaN`, `.sum()` returns `0`, boolean
comparisons against `NaN` return `False`, and `~` on the wrong dtype does
bitwise math instead of logical negation. None of these throw an error.
All of them produce a plausible-looking number. The only way to catch them
is to distrust "looks fine" and actually spot-check specific rows against
what you know to be true about the underlying data — which is exactly
what the `nsmallest()` / single-row `.T` checks throughout this session
kept turning up.

**Final verification** (post-fix, county `28099`, year `2024` — the row
that surfaced Bugs 3 and 4):

| Feature | Value | Correct? |
|---|---|---|
| `tmax_avg` | `NaN` | ✅ station has 0 TMAX readings |
| `tmin_avg` | `NaN` | ✅ same station, same reason |
| `gdd` | `NaN` | ✅ depends on TMAX |
| `heat_stress` | `NaN` | ✅ depends on TMAX |
| `prcp_total` | `455.4` | ✅ station's real precip data, preserved |
| `prcp_days` | `34.0` | ✅ same |

Dataset-wide, `gdd` and `heat_stress` null counts (35 each) now exactly
match `tmax_avg`/`tmin_avg` (35 each) — as they should, since both are
temperature-derived — while `prcp_total` carries its own, smaller,
independent null count (25). That divergence is the signal that each
feature is finally being evaluated on its own evidence rather than
inheriting another feature's gaps.
