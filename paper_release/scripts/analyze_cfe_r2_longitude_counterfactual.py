#!/usr/bin/env python3
"""What sets the global-reach conditional waveform residual (~12.7%)?

Decomposes the residual into a SPATIAL (longitudinal distribution of receiving
capacity) and a TEMPORAL (clean-supply waveform shape) contribution, holding the
two-pass waveform-matching procedure (Methods 4.4 / SI S10) fixed at the floor
corner (phi = 1 fully routable, tau = 500 ms global reach).

Experiment 1 - LONGITUDE counterfactual
  Replace the 40 real hyperscaler partners with synthetic receiving partners
  spaced uniformly in longitude, each carrying the capacity-weighted mean clean
  shape of the real partners re-phased to its longitude (a near-noon partner at
  every longitude; the mid-Pacific receiving void is filled). Also roll a single
  high-quality solar partner (Chile) to all longitudes.
    -> the residual barely moves (12.7% -> 13.0/13.2%, roll 11.5%): it is NOT
       caused by longitudinal clustering / the Pacific void.

Experiment 2 - SUPPLY-WAVEFORM counterfactual (uniform longitude, vary only shape)
  pure solar (zero at night) / real solar+wind / wind-heavier dictionaries.
    -> the residual moves ~4-5 pp (10.7 / 13.0 / 15.1%): it is a waveform-shape
       (phase) property. Consistent with ~50% of the residual falling in the
       local 18:00-23:00 evening window.

Result: the 12.7% conditional residual is set by the timing-shape mismatch between evening-peaked
demand and daytime-dominated clean supply (human routines vs Earth's rotation),
essentially invariant to where receiving capacity sits in longitude.

Reproduces the numbers cited in Supplementary Section S15.2.
"""
from __future__ import annotations
import numpy as np
import analyze_cfe_geographic_portfolio_ai as base
import analyze_cfe_global_sites as G
import analyze_cfe_global_sites_stations as ST
import analyze_cfe_r2_waveform_routing as WF


def load():
    _, demand, _, _, smeta = G.load_inputs_expanded()
    idx = demand[next(iter(demand))]["d"].index
    utc_h = np.asarray(idx.hour, dtype=float)
    station_curves, station_meta = ST.load_stations(idx, smeta)
    S_home, shape_home = {}, {}
    for c in demand:
        pool = ST.station_pool(c, "L0", station_meta, smeta)
        S, _ = ST.station_basis(station_curves, pool, idx)
        if getattr(S, "size", 0):
            S_home[c], shape_home[c] = S, WF.national_shape(S)
        else:
            S_home[c] = shape_home[c] = None
    countries = [c for c in demand if S_home.get(c) is not None and c in smeta]
    reach = WF.precompute_reach(countries, smeta)
    resid, dvec, lhr = {}, {}, {}
    for c in countries:
        d = demand[c]["d"].to_numpy(float)
        S1 = base.equal_energy_align(S_home[c], float(d.mean()))
        _, _, fit = base.portfolio_nnls(d, S1)
        resid[c] = np.maximum(d - fit, 0.0)
        dvec[c] = d
        lhr[c] = ((utc_h + smeta[c]["lon"] / 15.0) % 24).astype(int)
    return countries, smeta, shape_home, reach, resid, dvec, lhr, utc_h


def uncovered(c, P, resid, dvec):
    d, r = dvec[c], resid[c]
    if P is None or P.shape[1] == 0 or r.sum() <= 0:
        return r
    P1 = base.equal_energy_align(P, float(r.mean()))
    _, _, fr = base.portfolio_nnls(r, P1)
    return r - np.minimum(r, np.maximum(fr, 0.0))


def med_floor(countries, partner_fn, resid, dvec):
    v = [uncovered(c, partner_fn(c), resid, dvec).sum() / max(dvec[c].sum(), 1e-12)
         for c in countries]
    return float(np.median(v)) * 100


def canonical_shape(countries, smeta, shape_home, utc_h):
    hy = [r for r in countries if r in G.HYPERSCALER_EXTENDED and shape_home.get(r) is not None]
    prof, cnt = np.zeros(24), np.zeros(24)
    for r in hy:
        slh = ((utc_h + smeta[r]["lon"] / 15.0) % 24).astype(int)
        for h in range(24):
            m = slh == h
            if m.any():
                prof[h] += shape_home[r][m].mean(); cnt[h] += 1
    g = prof / np.maximum(cnt, 1)
    return g / g.mean()


def main():
    countries, smeta, shape_home, reach, resid, dvec, lhr, utc_h = load()
    print(f"usable countries = {len(countries)}")
    g = canonical_shape(countries, smeta, shape_home, utc_h)

    def synth(gshape, lam):
        return gshape[((utc_h + lam / 15.0) % 24).astype(int)]

    def uniform(gshape, step):
        P = np.stack([synth(gshape, l) for l in np.arange(-180, 180, step)], axis=1)
        return lambda c: P

    def baseline(c):
        parts = [r for r in reach[c][500] if shape_home.get(r) is not None]
        return np.stack([shape_home[r] for r in parts], axis=1) if parts else None

    ref = next(c for c in ("CL", "ZA", "AU", "ES") if shape_home.get(c) is not None)
    rl, rs = smeta[ref]["lon"], shape_home[ref]

    def roll(step):
        cols = [np.roll(rs, -int(round((l - rl) / 15.0))) for l in np.arange(-180, 180, step)]
        P = np.stack(cols, axis=1)
        return lambda c: P

    print("\n=== Experiment 1: LONGITUDE (median conditional residual, phi=1, tau=500) ===")
    b = med_floor(countries, baseline, resid, dvec)
    print(f"  baseline (real 40 partners)     = {b:5.2f}%")
    print(f"  uniform longitude, 15 deg       = {med_floor(countries, uniform(g,15), resid, dvec):5.2f}%")
    print(f"  uniform longitude, 30 deg       = {med_floor(countries, uniform(g,30), resid, dvec):5.2f}%")
    print(f"  single solar ({ref}) at all lon   = {med_floor(countries, roll(15), resid, dvec):5.2f}%")

    solar = np.clip(np.sin((np.arange(24) - 6) / 12.0 * np.pi), 0, None)
    g_solar = solar / solar.mean()
    g_wind = (g + 0.6); g_wind /= g_wind.mean()
    print("\n=== Experiment 2: SUPPLY WAVEFORM (uniform 15 deg longitude) ===")
    print(f"  pure solar (zero at night)      = {med_floor(countries, uniform(g_solar,15), resid, dvec):5.2f}%")
    print(f"  real solar+wind                 = {med_floor(countries, uniform(g,15), resid, dvec):5.2f}%")
    print(f"  wind-heavier (higher night)     = {med_floor(countries, uniform(g_wind,15), resid, dvec):5.2f}%")

    by_h, tot = np.zeros(24), 0.0
    for c in countries:
        u = uncovered(c, baseline(c), resid, dvec)
        for h in range(24):
            by_h[h] += u[lhr[c] == h].sum()
        tot += u.sum()
    p = by_h / tot * 100
    print("\n=== residual by local hour (baseline) ===")
    print(f"  evening 18-23 = {p[18:24].sum():.0f}%   deep-night 00-05 = {p[:6].sum():.0f}%   day 06-17 = {p[6:18].sum():.0f}%")


if __name__ == "__main__":
    main()
