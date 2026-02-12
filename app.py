import io
import re
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

# Global constant: total assets column name
TOTAL_ASSETS_COL = "Total Assets (EUR bn)"

# Fallback SRT cost used by the model (input CSV values are ignored; overridden via UI slider).
DEFAULT_SRT_COST_PCT = 0.2  # percent (0.2% = 20 bps)


import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
# ============================================================
# Transition matrices + greedy allocator (integrated)
# ============================================================

# --- Segment B: donors (B1) -> receivers (B2) ---
B1_DONORS = ['B1_SME_CORE', 'B1_MIDCORP_CORE', 'B1_TRADE_FIN_CORE', 'B1_CRE_CORE', 'B1_CONSUMER_FIN_CORE']
B2_RECEIVERS = ['B2_SME_RISK_UP', 'B2_MIDCORP_RISK_UP', 'B2_STRUCTURED_WC', 'B2_CRE_RISK_UP', 'B2_SPECIALTY_ABL']


# Allowed receiver buckets per donor (drives eligibility; UI enforces the same constraint)
ALLOWED_RECEIVERS_BY_DONOR: Dict[str, set] = {
    # SME core: allow all receivers
    "B1_SME_CORE": set(B2_RECEIVERS),
    # Mid-corp core: allow all receivers
    "B1_MIDCORP_CORE": set(B2_RECEIVERS),
    # Trade finance core: allow all receivers
    "B1_TRADE_FIN_CORE": set(B2_RECEIVERS),
    # CRE core: allow all receivers
    "B1_CRE_CORE": set(B2_RECEIVERS),
    # Consumer finance core: allow all receivers (no hard disablement)
    "B1_CONSUMER_FIN_CORE": set(B2_RECEIVERS),
}

# ΔNet spread (bps) for B (positive = increases net margin)
DELTA_SPREAD_BPS_B: Dict[Tuple[str, str], float] = {
    ('B1_SME_CORE', 'B2_SME_RISK_UP'): +225.000,
    ('B1_SME_CORE', 'B2_MIDCORP_RISK_UP'): +175.000,
    ('B1_SME_CORE', 'B2_STRUCTURED_WC'): +75.000,
    ('B1_SME_CORE', 'B2_CRE_RISK_UP'): +95.000,
    ('B1_MIDCORP_CORE', 'B2_SME_RISK_UP'): +270.000,
    ('B1_MIDCORP_CORE', 'B2_MIDCORP_RISK_UP'): +220.000,
    ('B1_MIDCORP_CORE', 'B2_STRUCTURED_WC'): +120.000,
    ('B1_MIDCORP_CORE', 'B2_CRE_RISK_UP'): +140.000,
    ('B1_MIDCORP_CORE', 'B2_SPECIALTY_ABL'): +320.000,
    ('B1_TRADE_FIN_CORE', 'B2_MIDCORP_RISK_UP'): +280.000,
    ('B1_TRADE_FIN_CORE', 'B2_STRUCTURED_WC'): +180.000,
    ('B1_CRE_CORE', 'B2_MIDCORP_RISK_UP'): +230.000,
    ('B1_CRE_CORE', 'B2_STRUCTURED_WC'): +130.000,
    ('B1_CRE_CORE', 'B2_CRE_RISK_UP'): +150.000,
    # Consumer finance core margin updated to 300 bps (was 350 bps).
    # Since receiver absolute spreads are unchanged, increase deltas accordingly.
    ('B1_CONSUMER_FIN_CORE', 'B2_SME_RISK_UP'): +150.000,
    ('B1_CONSUMER_FIN_CORE', 'B2_MIDCORP_RISK_UP'): +100.000,
    ('B1_SME_CORE', 'B2_SPECIALTY_ABL'): +275.000,
    ('B1_TRADE_FIN_CORE', 'B2_SME_RISK_UP'): +330.000,
    ('B1_TRADE_FIN_CORE', 'B2_CRE_RISK_UP'): +200.000,
    ('B1_TRADE_FIN_CORE', 'B2_SPECIALTY_ABL'): +380.000,
    ('B1_CRE_CORE', 'B2_SME_RISK_UP'): +280.000,
    ('B1_CRE_CORE', 'B2_SPECIALTY_ABL'): +330.000,

}



# Absolute net spread (bps) by receiver (same for all donors)
ABS_NET_SPREAD_BPS_BY_B2: Dict[str, float] = {
    'B2_SME_RISK_UP': 450,
    'B2_MIDCORP_RISK_UP': 400,
    'B2_STRUCTURED_WC': 300,
    'B2_CRE_RISK_UP': 320,
    'B2_SPECIALTY_ABL': 500,
}


# Absolute net spread (bps) by donor (core book) — inferred / calibrated.
# Used to compute lost NII on sold flow (whole-loan forward flow).
DONOR_ABS_NET_SPREAD_BPS_BY_B1: Dict[str, float] = {
    "B1_SME_CORE": 225.0,
    "B1_MIDCORP_CORE": 180.0,
    "B1_TRADE_FIN_CORE": 120.0,
    "B1_CRE_CORE": 170.0,
    "B1_CONSUMER_FIN_CORE": 300.0,
}


@dataclass(frozen=True)
class Cell:
    donor: str
    receiver: str
    delta_rwa_pp: float  # ΔRWA density (pp), negative means RWAs fall per unit exposure moved
    donor_risk_weight: float  # decimal, e.g. 0.90
    receiver_risk_weight: float  # decimal, e.g. 0.30
    abs_net_spread_bps: float  # receiver absolute net spread (bps)
    delta_s_eff_dec: float  # effective spread-like term (decimal), per new formula
    ratio: float  # delta_s_eff_dec per unit of RWA reduction (profitability per RWA)

@dataclass
class Allocation:
    donor: str
    receiver: str
    exposure_used_eur_bn: float
    rwa_reduction_eur_bn: float
    assets_redeploy_used_eur_bn: float
    donor_risk_weight: float
    receiver_risk_weight: float
    delta_rwa_pp: float
    abs_net_spread_bps: float
    delta_s_eff_dec: float
    ratio: float


# Risk weights used for the transition engine
DONOR_RISK_WEIGHT: Dict[str, float] = {
    # Updated donor risk weights per user calibration
    'B1_SME_CORE': 0.850,
    'B1_MIDCORP_CORE': 0.650,
    'B1_TRADE_FIN_CORE': 0.400,
    'B1_CRE_CORE': 0.800,
    'B1_CONSUMER_FIN_CORE': 0.750,
}

RECEIVER_RISK_WEIGHT: Dict[str, float] = {
    'B2_SME_RISK_UP': 0.600,
    'B2_MIDCORP_RISK_UP': 0.550,
    'B2_STRUCTURED_WC': 0.350,
    'B2_CRE_RISK_UP': 0.500,
    'B2_SPECIALTY_ABL': 0.450,
}



def transition_eligibility_product(
    donor: str,
    receiver: str,
    *,
    donor_net_margin_bps_by_b1: Dict[str, float] = DONOR_ABS_NET_SPREAD_BPS_BY_B1,
    receiver_net_margin_bps_by_b2: Dict[str, float] = ABS_NET_SPREAD_BPS_BY_B2,
    donor_risk_weight_by_b1: Dict[str, float] = DONOR_RISK_WEIGHT,
) -> float:
    """Eligibility product used to decide whether a donor -> receiver transition is selectable.

    Steps (as specified):
      1) receiver_net_margin / donor_net_margin
      2) donor_risk_weight / receiver_net_margin
      3) product = step1 * step2
      4) eligible if product > 0

    Notes:
      - Net margins are taken from the absolute net spread calibration (bps).
      - If any required denominator is 0 (or missing), the product is treated as 0 (non-eligible).
    """
    donor_nm = float(donor_net_margin_bps_by_b1.get(donor, 0.0) or 0.0)
    recv_nm = float(receiver_net_margin_bps_by_b2.get(receiver, 0.0) or 0.0)
    donor_rw = float(donor_risk_weight_by_b1.get(donor, 0.0) or 0.0)

    if donor_nm == 0.0 or recv_nm == 0.0:
        return 0.0

    q1 = recv_nm / donor_nm
    q2 = donor_rw / recv_nm
    return q1 * q2


def _eligible_cells_by_donor(
    donors: List[str],
    sale_share: float,
    delta_spread_bps: Dict[Tuple[str, str], float],
    retained_risk_dec: float = 0.0,
    donor_roll_pct_by_donor: Optional[Dict[str, float]] = None,
    receiver_split_by_donor: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, List[Cell]]:
    """Return all *eligible* receiver cells per donor.

    **Eligibility** is controlled purely by `receiver_split_by_donor` (share > 0),
    subject to the hard guardrails in `ALLOWED_RECEIVERS_BY_DONOR`.

    We still compute `delta_s_eff_dec` and `ratio` for reporting/auditing.

    Key change:
      - The legacy ΔRWA matrix (DELTA_RWA_PP_B) is no longer used.
      - The RWA-efficiency factor used in the ROE math is the quotient:
            donor_risk_weight / receiver_risk_weight
        for each donor -> receiver transition.
    """
    cells: Dict[str, List[Cell]] = {d: [] for d in donors}

    sale_share = float(sale_share)
    retained_risk_dec = max(min(float(retained_risk_dec or 0.0), 1.0), 0.0)
    eff = sale_share * (1.0 - retained_risk_dec)

    for d in donors:
        donor_rw = float(DONOR_RISK_WEIGHT.get(d, 0.0))
        if donor_rw <= 0:
            continue

        allowed_receivers = ALLOWED_RECEIVERS_BY_DONOR.get(d, set(B2_RECEIVERS))
        for r in allowed_receivers:
            receiver_rw = float(RECEIVER_RISK_WEIGHT.get(r, 0.0))
            if receiver_rw <= 0:
                continue

            # Eligibility gate:
            #   - transition must have a positive eligibility product (per user rule-set)
            #   - and (if provided) the user-defined receiver split must be > 0
            if transition_eligibility_product(d, r) <= 0.0:
                continue
            if receiver_split_by_donor is not None:
                if float(receiver_split_by_donor.get(d, {}).get(r, 0.0) or 0.0) <= 0.0:
                    continue

            # Need spread calibration for reporting/profit computations
            dspr = delta_spread_bps.get((d, r))
            if dspr is None:
                continue

            abs_spread_bps = float(ABS_NET_SPREAD_BPS_BY_B2.get(r, 0.0))

            # RWA-efficiency factor (replaces DELTA_RWA_PP_B usage)
            delta_rwa_density = donor_rw / receiver_rw

            # Economics term: redeployment value proxy (keeps legacy structure: scales with
            # RWA-efficiency factor, net relief efficiency and receiver absolute spread)
            delta_s_eff_bps = delta_rwa_density * eff * abs_spread_bps
            delta_s_eff_dec = delta_s_eff_bps / 10000.0

            rwa_red_per_eur = donor_rw * eff
            ratio = (delta_s_eff_dec / rwa_red_per_eur) if rwa_red_per_eur > 0 else 0.0

            cells[d].append(
                Cell(
                    donor=d,
                    receiver=r,
                    delta_rwa_pp=float("nan"),  # legacy field retained for schema compatibility
                    donor_risk_weight=donor_rw,
                    receiver_risk_weight=receiver_rw,
                    abs_net_spread_bps=abs_spread_bps,
                    delta_s_eff_dec=float(delta_s_eff_dec),
                    ratio=float(ratio),
                )
            )

    # drop donors with no eligible receivers
    return {d: lst for d, lst in cells.items() if lst}


def allocate_rwa_reduction_equal_receivers(
    rwa_target_eur_bn: float,
    donor_exposure_eur_bn: Dict[str, float],
    sale_share: float,
    retained_risk_dec: float = 0.0,
    receiver_split_by_donor: Optional[Dict[str, Dict[str, float]]] = None,
    donor_roll_pct_by_donor: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Allocator (Segment B only) with *equal split across eligible receivers*.

    Changes vs earlier greedy version:
      - No prioritization across donors by profitability.
      - For each donor, exposure used is split equally across all eligible receivers.

    RWA reduction achieved per allocation line:
        rwa_reduction = exposure_used * donor_risk_weight * srt_efficiency

    Assets redeployed used per allocation line (receiver-side):
        assets_redeploy = rwa_reduction / receiver_risk_weight
    """
    donors = [d for d in B1_DONORS if donor_exposure_eur_bn.get(d, 0.0) > 0]

    cells_by_donor = _eligible_cells_by_donor(
        donors=donors,
        sale_share=float(sale_share),
        delta_spread_bps=DELTA_SPREAD_BPS_B,
        retained_risk_dec=retained_risk_dec,
        donor_roll_pct_by_donor=donor_roll_pct_by_donor,
        receiver_split_by_donor=receiver_split_by_donor,
    )

    if not cells_by_donor:
        return {
            "allocations": [],
            "total_rwa_reduction_eur_bn": 0.0,
            "total_exposure_used_eur_bn": 0.0,
            "total_assets_redeploy_used_eur_bn": 0.0,
            "remaining_rwa_target_eur_bn": float(rwa_target_eur_bn),
            "ranked_donors": [],
            "status": "NO_ELIGIBLE_TRANSITIONS",
        }

    remaining = float(rwa_target_eur_bn)
    allocs: List[Allocation] = []
    total_rwa_red = 0.0
    total_expo = 0.0
    total_assets_redeploy = 0.0

    retained_risk_dec = max(min(float(retained_risk_dec or 0.0), 1.0), 0.0)

    eff = float(sale_share) * (1.0 - retained_risk_dec)

    # No prioritization: iterate donors in fixed order
    for d in B1_DONORS:
        if remaining <= 0:
            break
        if d not in cells_by_donor:
            continue

        expo_avail = float(donor_exposure_eur_bn.get(d, 0.0))
        if expo_avail <= 0:
            continue

        # donor RWA reduction per EUR exposure moved
        donor_rw = float(DONOR_RISK_WEIGHT.get(d, 0.0))
        red_per_eur = donor_rw * eff
        if red_per_eur <= 0:
            continue

        expo_needed = remaining / red_per_eur
        expo_used_total = min(expo_avail, expo_needed)
        if expo_used_total <= 0:
            continue

        # Split across eligible receivers (user-defined weights if provided; otherwise equal)
        cells = cells_by_donor[d]

        # Build receiver weights map for this donor
        w_map = {}
        if receiver_split_by_donor and isinstance(receiver_split_by_donor, dict):
            w_map = receiver_split_by_donor.get(d, {}) or {}

        # Filter weights to eligible receivers (cell[0] is receiver id) and positive values
        weights = []
        receivers = []
        for cell_obj in cells:
            r = cell_obj.receiver
            w = w_map.get(r, None)
            if w is None:
                continue
            try:
                wv = float(w)
            except Exception:
                continue
            if wv > 0:
                receivers.append(r)
                weights.append(wv)

        if len(weights) == 0:
            # fallback: equal split across eligible receivers
            receivers = [c.receiver for c in cells]
            weights = [1.0] * len(receivers)

        # Normalize
        w_sum = float(sum(weights)) if weights else 0.0
        if w_sum <= 0:
            receivers = [c.receiver for c in cells]
            weights = [1.0] * len(receivers)
            w_sum = float(sum(weights))

        w_norm = {r: (w / w_sum) for r, w in zip(receivers, weights)}

        for cell in cells:
            r = cell.receiver
            expo_used = expo_used_total * float(w_norm.get(r, 0.0))
            rwa_red = expo_used * red_per_eur
            assets_redeploy = rwa_red / float(cell.receiver_risk_weight)

            allocs.append(
                Allocation(
                    donor=cell.donor,
                    receiver=cell.receiver,
                    exposure_used_eur_bn=expo_used,
                    rwa_reduction_eur_bn=rwa_red,
                    assets_redeploy_used_eur_bn=assets_redeploy,
                    donor_risk_weight=cell.donor_risk_weight,
                    receiver_risk_weight=cell.receiver_risk_weight,
                    delta_rwa_pp=cell.delta_rwa_pp,
                    abs_net_spread_bps=cell.abs_net_spread_bps,
                    delta_s_eff_dec=cell.delta_s_eff_dec,
                    ratio=cell.ratio,
                )
            )

            total_expo += expo_used
            total_rwa_red += rwa_red
            total_assets_redeploy += assets_redeploy

        remaining -= expo_used_total * red_per_eur

    status = "OK" if remaining <= 1e-9 else "TARGET_NOT_MET"
    return {
        "allocations": allocs,
        "total_rwa_reduction_eur_bn": total_rwa_red,
        "total_exposure_used_eur_bn": total_expo,
        "total_assets_redeploy_used_eur_bn": total_assets_redeploy,
        "remaining_rwa_target_eur_bn": max(0.0, remaining),
        # kept key name for UI compatibility; now it's just donor order with receiver count
        "ranked_donors": [(d, "MULTI", len(cells_by_donor[d])) for d in B1_DONORS if d in cells_by_donor],
        "status": status,
    }

def allocate_until_profit_target(
    profit_target_eur_bn_yr: float,
    donor_exposure_eur_bn: Dict[str, float],
    sale_share: float,
    retained_risk_dec: float = 0.0,
    tax_dec: float = 0.0,
    receiver_split_by_donor: Optional[Dict[str, Dict[str, float]]] = None,
    donor_roll_pct_by_donor: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Allocator (Segment B) that allocates donor exposure until a *profit* target is met.

    - Uses the same donor->receiver eligibility logic as `allocate_rwa_reduction_equal_receivers`
      (ΔRWA < 0 and ΔSpread > 0, after SRT-cost adjustment).
    - Uses the user-provided receiver split matrix (per donor) when available; otherwise falls back to equal split.
    - Prioritizes donors by *profit per EUR exposure* (weighted-average across the eligible receivers for that donor
      under the receiver split).

    Returns allocations plus achieved profit / RWA reduction / exposure used.
    """
    profit_target = float(profit_target_eur_bn_yr) if profit_target_eur_bn_yr is not None else 0.0
    if not np.isfinite(profit_target) or profit_target <= 0:
        return {
            "allocations": [],
            "total_profit_eur_bn_yr": 0.0,
            "total_rwa_reduction_eur_bn": 0.0,
            "total_exposure_used_eur_bn": 0.0,
            "total_assets_redeploy_used_eur_bn": 0.0,
            "remaining_profit_target_eur_bn_yr": max(0.0, profit_target),
            "ranked_donors": [],
            "status": "NO_PROFIT_TARGET",
        }

    donors = [d for d in B1_DONORS if donor_exposure_eur_bn.get(d, 0.0) > 0]

    cells_by_donor = _eligible_cells_by_donor(
        donors=donors,
        sale_share=float(sale_share),
        delta_spread_bps=DELTA_SPREAD_BPS_B,
        retained_risk_dec=retained_risk_dec,
        donor_roll_pct_by_donor=donor_roll_pct_by_donor,
        receiver_split_by_donor=receiver_split_by_donor,
    )

    if not cells_by_donor:
        return {
            "allocations": [],
            "total_profit_eur_bn_yr": 0.0,
            "total_rwa_reduction_eur_bn": 0.0,
            "total_exposure_used_eur_bn": 0.0,
            "total_assets_redeploy_used_eur_bn": 0.0,
            "remaining_profit_target_eur_bn_yr": profit_target,
            "ranked_donors": [],
            "status": "NO_ELIGIBLE_TRANSITIONS",
        }

    retained_risk_dec = max(min(float(retained_risk_dec or 0.0), 1.0), 0.0)

    eff = float(sale_share) * (1.0 - retained_risk_dec)
    tx = float(tax_dec) if np.isfinite(float(tax_dec)) else 0.0
    tx = max(min(tx, 1.0), 0.0)

    donor_rank = []
    donor_receiver_weights: Dict[str, Dict[str, float]] = {}
    donor_profit_per_expo: Dict[str, float] = {}

    # Pre-compute receiver weights + weighted-average profit per EUR exposure for each donor
    for d, cells in cells_by_donor.items():
        w_map = (receiver_split_by_donor or {}).get(d, {}) if isinstance(receiver_split_by_donor, dict) else {}
        receivers, weights = [], []
        for c in cells:
            r = c.receiver
            w = w_map.get(r, None)
            if w is None:
                continue
            try:
                wv = float(w)
            except Exception:
                continue
            if wv > 0:
                receivers.append(r)
                weights.append(wv)

        if len(weights) == 0:
            receivers = [c.receiver for c in cells]
            weights = [1.0] * len(receivers)

        w_sum = float(sum(weights)) if weights else 0.0
        if w_sum <= 0:
            receivers = [c.receiver for c in cells]
            weights = [1.0] * len(receivers)
            w_sum = float(sum(weights))

        w_norm = {r: (w / w_sum) for r, w in zip(receivers, weights)}
        donor_receiver_weights[d] = w_norm

        avg_delta_s_eff = 0.0
        for c in cells:
            avg_delta_s_eff += float(w_norm.get(c.receiver, 0.0)) * float(c.delta_s_eff_dec)

        profit_per_expo = avg_delta_s_eff * (1.0 - tx)
        donor_profit_per_expo[d] = float(profit_per_expo)
        donor_rank.append((d, float(profit_per_expo), len(cells)))

    # IMPORTANT: consume donors in the fixed, user-expected order:
    # SME term -> Mid-corp -> EM corp -> CRE Non-HV-CRE
    remaining_profit = profit_target
    allocs: List[Allocation] = []
    total_profit = 0.0
    total_rwa_red = 0.0
    total_expo = 0.0
    total_assets_redeploy = 0.0

    for d in B1_DONORS:
        if remaining_profit <= 1e-12:
            break
        if d not in cells_by_donor:
            continue

        profit_per_expo = float(donor_profit_per_expo.get(d, 0.0))
        if profit_per_expo <= 0:
            continue

        expo_avail = float(donor_exposure_eur_bn.get(d, 0.0))
        if expo_avail <= 0:
            continue

        expo_needed = remaining_profit / profit_per_expo
        expo_used_total = min(expo_avail, expo_needed)
        if expo_used_total <= 0:
            continue

        donor_rw = float(DONOR_RISK_WEIGHT.get(d, 0.0))
        red_per_eur = donor_rw * eff
        if red_per_eur <= 0:
            continue

        w_norm = donor_receiver_weights.get(d, {})
        cells = cells_by_donor[d]

        for cell in cells:
            r = cell.receiver
            w = float(w_norm.get(r, 0.0))
            if w <= 0:
                continue

            expo_used = expo_used_total * w
            rwa_red = expo_used * red_per_eur
            assets_redeploy = rwa_red / float(cell.receiver_risk_weight)
            contrib = expo_used * float(cell.delta_s_eff_dec) * (1.0 - tx)

            allocs.append(
                Allocation(
                    donor=cell.donor,
                    receiver=cell.receiver,
                    exposure_used_eur_bn=expo_used,
                    rwa_reduction_eur_bn=rwa_red,
                    assets_redeploy_used_eur_bn=assets_redeploy,
                    donor_risk_weight=cell.donor_risk_weight,
                    receiver_risk_weight=cell.receiver_risk_weight,
                    delta_rwa_pp=cell.delta_rwa_pp,
                    abs_net_spread_bps=cell.abs_net_spread_bps,
                    delta_s_eff_dec=cell.delta_s_eff_dec,
                    ratio=cell.ratio,
                )
            )

            total_expo += expo_used
            total_rwa_red += rwa_red
            total_assets_redeploy += assets_redeploy
            total_profit += contrib

        remaining_profit = max(0.0, profit_target - total_profit)

    status = "OK" if remaining_profit <= 1e-9 else "TARGET_NOT_MET"
    return {
        "allocations": allocs,
        "total_profit_eur_bn_yr": total_profit,
        "total_rwa_reduction_eur_bn": total_rwa_red,
        "total_exposure_used_eur_bn": total_expo,
        "total_assets_redeploy_used_eur_bn": total_assets_redeploy,
        "remaining_profit_target_eur_bn_yr": remaining_profit,
        "ranked_donors": [(d, float(ppe), int(n)) for d, ppe, n in donor_rank],
        "status": status,
    }


# Default donor exposure split (since CSV has no subsegment exposures)
DEFAULT_DONOR_SPLIT_OF_TOTAL_ASSETS = {
    # Segment B donors (share of total assets, if not provided by input file)
    'B1_SME_CORE': 0.060,
    'B1_MIDCORP_CORE': 0.050,
    'B1_TRADE_FIN_CORE': 0.030,
    'B1_CRE_CORE': 0.040,
    'B1_CONSUMER_FIN_CORE': 0.030,
}


# Optional per-bank donor split columns in the input CSV (percent of total assets).
# If present, these override the default donor split *per bank* (missing values fall back to defaults).
DONOR_SPLIT_COLS_PCT = {
    'B1_SME_CORE': 'SME_term',
    'B1_MIDCORP_CORE': 'MidCorp_nonIG',
    'B1_TRADE_FIN_CORE': 'Trade_finance',
    'B1_CRE_CORE': 'CRE_non_HVCRE',
    'B1_CONSUMER_FIN_CORE': 'Consumer_finance',
}

def donor_split_from_row(row: pd.Series, default_split: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Return donor split (fractions of total assets) for one bank row.

    - Uses DEFAULT_DONOR_SPLIT_OF_TOTAL_ASSETS unless overridden by CSV % columns.
    - Each CSV % value is interpreted as percent (e.g., 6 -> 6% -> 0.06).
    - Missing/NaN/negative values fall back to defaults for that donor.
    """
    base = dict(default_split or DEFAULT_DONOR_SPLIT_OF_TOTAL_ASSETS)
    for donor, col in DONOR_SPLIT_COLS_PCT.items():
        if col in row.index:
            v = row.get(col)
            try:
                v = float(v)
            except Exception:
                v = np.nan
            if np.isfinite(v) and v >= 0:
                base[donor] = v / 100.0
    return base


def donor_eligible_exposure_long(
    banks_df: pd.DataFrame,
    donor_roll_pct_by_donor: Optional[Dict[str, float]] = None,
    donor_split_override_by_bank: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Return long df of *annual* eligible donor exposure flow per bank (EUR bn / yr).

    Flow eligible exposure = Total Assets * donor split * rolling fraction (percent).
    Rolling fractions are in percent (e.g. 15 -> 15% / yr). Missing donors default to 15%.
    """
    rows: list[dict] = []
    for _, r in banks_df.iterrows():
        bank = str(r.get("Bank", ""))
        if not bank or bank == "nan":
            bank = str(r.get("Bank Name", ""))
        ta = float(r.get("Total Assets (EUR bn)", np.nan))
        if not np.isfinite(ta):
            continue
        split = donor_split_from_row(r, DEFAULT_DONOR_SPLIT_OF_TOTAL_ASSETS)
        # apply optional override dict by bank name
        if donor_split_override_by_bank and bank in donor_split_override_by_bank:
            for d, v in donor_split_override_by_bank[bank].items():
                try:
                    split[d] = float(v)
                except Exception:
                    pass
        for donor, w in split.items():
            try:
                w_f = float(w)
            except Exception:
                continue
            if not np.isfinite(w_f) or w_f <= 0:
                continue
            expo = ta * w_f
            roll_pct = 15.0
            if donor_roll_pct_by_donor and donor in donor_roll_pct_by_donor:
                try:
                    roll_pct = float(donor_roll_pct_by_donor.get(donor, 15.0))
                except Exception:
                    roll_pct = 15.0
            if np.isfinite(roll_pct):
                roll_pct = max(min(roll_pct, 25.0), 0.0)
            else:
                roll_pct = 15.0
            expo_elig = expo * roll_pct / 100.0
            rows.append({"Bank": bank, "Donor": donor, "Eligible_Exposure_EUR_bn": expo_elig})
    return pd.DataFrame(rows)



def compute_max_roe_uplift_map(
    sim_df: pd.DataFrame,
    banks_sel: pd.DataFrame,
    sale_share_pct: float,
    retained_risk_pct: float = 0.0,
    gain_on_sale_bps: float = 0.0,
    servicing_fee_bps: float = 0.0,
    origination_fee_bps: float = 0.0,
    override_tax_rate: float | None = None,
    donor_roll_pct_by_donor: Optional[Dict[str, float]] = None,
    receiver_split_by_donor: Optional[Dict[str, Dict[str, float]]] = None,
    tol_pct: float = 0.5,
) -> Dict[str, float]:
    """Max annual ROE uplift (bp p.a.) per bank when *all* eligible donor capacity is used for redeployment.

    Implementation mirrors the former 'Max. ROE uplift (bp)' column in the table under chart (3),
    but returns a dict so it can be used for visualization (chart 2).
    """
    try:
        if sim_df is None or sim_df.empty or banks_sel is None or banks_sel.empty:
            return {}

        sim_df_max = sim_df.copy()
        # Oversize the RWA reduction targets so the allocator becomes capacity-constrained.
        sim_df_max["Effective_RWA_Reduction_EUR_bn_Yr"] = 1e12
        sim_df_max["Gross_RWA_Offload_EUR_bn_Yr"] = 1e12

        roe_df_maxcap = compute_roe_delta_transitions_greedy(
            sim_df_max,
            banks_sel,
            roe_target_bp=1e6,
            sale_share_pct=sale_share_pct,
            retained_risk_pct=retained_risk_pct,

            gain_on_sale_bps=gain_on_sale_bps,
            servicing_fee_bps=servicing_fee_bps,
            origination_fee_bps=origination_fee_bps,  # very high target -> exhaust redeployment capacity
            apply_roe_target=True,
            
            override_tax_rate=override_tax_rate,
            require_exact_target=False,
            target_tolerance_pct=float(tol_pct),
            donor_roll_pct_by_donor=donor_roll_pct_by_donor,
            receiver_split_by_donor=receiver_split_by_donor,
        )

        if not isinstance(roe_df_maxcap, pd.DataFrame) or roe_df_maxcap.empty:
            return {}

        return (
            roe_df_maxcap.groupby("Bank", dropna=False)["ROE_delta_bp"]
            .max()
            .astype(float)
            .to_dict()
        )
    except Exception:
        return {}

def build_donor_exposures_from_total_assets(
    total_assets_eur_bn: float,
    donor_split: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    split = donor_split or DEFAULT_DONOR_SPLIT_OF_TOTAL_ASSETS
    return {k: float(total_assets_eur_bn) * float(v) for k, v in split.items()}


# ============================================================
# ---------------- Helpers (legacy) ----------------
# ============================================================
def rwa_to_assets(rwa_eur_bn: float, rwa_density_pct: float) -> float:
    density = max(float(rwa_density_pct), 1e-6) / 100.0
    return float(rwa_eur_bn) / density


def simulate_offload(banks_df: pd.DataFrame, scenarios_bps: dict, srt_efficiencies: List[float]) -> pd.DataFrame:
    """
    Per-bank version of the 'simple offload' module in **steady-state / annual** terms.

    Assumes CET1 capital is fixed: C = CET1_ratio * RWA.
    Removes the time-horizon dimension: all outputs are annual end-state requirements.
    """
    rows = []
    for _, b in banks_df.iterrows():
        bank = b["Bank Name"]
        R = float(b["Total RWA (EUR bn)"])
        cet1_ratio = float(b["CET1 Ratio (%)"]) / 100.0
        C = cet1_ratio * R
        d = float(b["RWA Density (%)"])

        for sc_name, adv_bps in scenarios_bps.items():
            delta = float(adv_bps) / 10000.0
            target = cet1_ratio + delta

            # R_eff = C / target, effective reduction = R - R_eff
            R_eff = C / target if target > 0 else np.nan
            eff_red = max(R - R_eff, 0.0) if np.isfinite(R_eff) else np.nan

            for eff in srt_efficiencies:
                eff = float(eff)
                gross_rwa = (eff_red / eff) if (eff > 0 and np.isfinite(eff_red)) else np.nan
                share = (gross_rwa / R) if (R > 0 and np.isfinite(gross_rwa)) else np.nan

                gross_ast = rwa_to_assets(gross_rwa, d) if np.isfinite(gross_rwa) else np.nan

                rows.append({
                    "Bank": bank,
                    "Country": b.get("Country", ""),
                    "Region": b.get("Region", ""),
                    "Reporting Period": b.get("Reporting Period", ""),
                    "Scenario": sc_name,
                    "US_CET1_Advantage_bps": float(adv_bps),
                    "SRT_Efficiency": f"{round(eff * 100):.0f}%",

                    "Current_CET1_Ratio_pct": round(cet1_ratio * 100, 2),
                    "Target_CET1_Ratio_pct": round(target * 100, 2),

                    "Total_RWA_EUR_bn": R,
                    "CET1_Capital_EUR_bn": round(C, 3),
                    "RWA_Density_pct": d,

                    # Steady-state / annual requirements
                    "Effective_RWA_Reduction_EUR_bn_Yr": round(eff_red, 3) if np.isfinite(eff_red) else np.nan,
                    "Gross_RWA_Offload_EUR_bn_Yr": round(gross_rwa, 3) if np.isfinite(gross_rwa) else np.nan,
                    "Gross_RWA_Offload_pct_of_RWA_Yr": round(share * 100, 3) if np.isfinite(share) else np.nan,
                    "Gross_Assets_Offloaded_EUR_bn_Yr": round(gross_ast, 3) if np.isfinite(gross_ast) else np.nan,
                })

    return pd.DataFrame(rows)


def compute_roe_delta_transitions_greedy(
    sim_df: pd.DataFrame,
    banks_df: pd.DataFrame,
    roe_target_bp: float,
    sale_share_pct: float,
    retained_risk_pct: float = 0.0,
    gain_on_sale_bps: float = 0.0,
    servicing_fee_bps: float = 0.0,
    origination_fee_bps: float = 0.0,
    apply_roe_target: bool = True,
    override_tax_rate: float | None = None,
    donor_split_override_by_bank: Optional[Dict[str, Dict[str, float]]] = None,
    donor_roll_pct_by_donor: Optional[Dict[str, float]] = None,
    receiver_split_by_donor: Optional[Dict[str, Dict[str, float]]] = None,
    require_exact_target: bool = False,
    target_tolerance_pct: float = 0.5,
) -> pd.DataFrame:
    """
    ΔROE module (transition-based).

    Workflow:
    - Take Gross_RWA_Offload (total horizon) and annualize it
    - Allocate annual RWA reduction across donor->receiver transitions (greedy)
      using only cells with (ΔRWA < 0 and ΔSpread > 0)
    - Convert each transition to profit using the corresponding Δspread (cell),
      and apply the legacy SRT-cost lever penalty logic
    - Sum profit contributions -> Addl_profit_Yr
    - Compute ROE uplift in legacy way: Addl_profit_Yr / CET1_Capital
    """
    bmap = banks_df.set_index("Bank Name")

    # Per-bank parameter maps
    # NOTE: SRT cost supplied via CSV is intentionally ignored (controlled via UI slider).
    tax_pct = bmap["Effective Tax Rate (%)"].to_dict()
    assets_total = bmap["Total Assets (EUR bn)"].to_dict()

    df = sim_df.copy()

    # Tax series (decimal)
    tx = (df["Bank"].map(tax_pct).fillna(0.0) / 100.0)
    if override_tax_rate is not None:
        tx = float(override_tax_rate)
    # Annual RWA reduction target for the transition engine:
    # Use *effective* RWA reduction (as in the original version before the gross-target change).
    gross = df["Gross_RWA_Offload_EUR_bn_Yr"].clip(lower=0)

    # Legacy lever term (gross/effective) kept so economics remain comparable
    eff = df["Effective_RWA_Reduction_EUR_bn_Yr"].clip(lower=0)
    df["gross_eff_lever"] = np.where(eff > 0, gross / eff, 1.0)
    lever_penalty = np.maximum(df["gross_eff_lever"] - 1.0, 0.0)
    eff_per_year = eff.to_numpy(dtype=float)  # STEADY-STATE FLOW: annual RWA reduction target equals maintained CET1 end-state level (no / Years)

    # outputs
    addl_profit_yr = []
    rwa_red_yr_achieved = []
    exposure_used_yr = []
    assets_redeploy_used_yr = []
    status_list = []
    struct_income_yr = []
    lost_nii_yr = []
    rwa_target_base_yr_list = []
    rwa_target_scaled_yr_list = []
    rwa_target_scale_list = []

    # audit trail (optional)
    audit_rows: List[Dict[str, object]] = []

    for i, row in df.iterrows():
        bank = row["Bank"]
        # Base (effective) annual RWA reduction target
        rwa_target_yr_base = float(eff_per_year[i]) if np.isfinite(eff_per_year[i]) else 0.0

        # Two-step allocation:
        #   Step 1 (Redeployment): allocate donor exposure until the *ROE / profit target* is met.
        #   Step 2 (CET1 uplift): allocate the RWAs required for CET1 uplift, *after* Step 1 has consumed capacity.
        #
        # Profit / ROE uplift is computed from Step 1 only.
        #
        # Base (annual effective) CET1-uplift requirement:
        rwa_target_cet1_yr = rwa_target_yr_base

        # ROE target (bp) -> annual profit target (EUR bn / yr)
        try:
            roe_target_bp_i = float(roe_target_bp)
        except Exception:
            roe_target_bp_i = 0.0
        if not np.isfinite(roe_target_bp_i):
            roe_target_bp_i = 0.0
        roe_target_bp_i = max(roe_target_bp_i, 0.0)

        cet1_cap_bn = float(row.get("CET1_Capital_EUR_bn", np.nan))
        if not np.isfinite(cet1_cap_bn) or cet1_cap_bn <= 0:
            profit_target_redeploy_yr = 0.0
        else:
            profit_target_redeploy_yr = (roe_target_bp_i / 10000.0) * cet1_cap_bn

        # Tax rate for this bank row (decimal, 0..1)
        tx_i = float(tx.iloc[i]) if hasattr(tx, "iloc") else float(tx)
        if not np.isfinite(tx_i):
            tx_i = 0.0
        tx_i = max(min(tx_i, 1.0), 0.0)

        # ------------------------------------------------------------
        # For reporting: Step 1 does not have an RWA target anymore (it's profit-driven).
        rwa_target_redeploy_yr = 0.0
        rwa_target_total_yr = rwa_target_cet1_yr

        # store targets for reporting (aligned with df rows)
        rwa_target_base_yr_list.append(rwa_target_yr_base)  # CET1 portion (base)
        rwa_target_scaled_yr_list.append(rwa_target_total_yr)  # placeholder; updated after Step 1
        rwa_target_scale_list.append(0.0)  # placeholder; updated after Step 1

        total_assets_bn = float(assets_total.get(bank, np.nan))
        if not np.isfinite(total_assets_bn) or total_assets_bn <= 0:
            addl_profit_yr.append(np.nan)
            rwa_red_yr_achieved.append(np.nan)
            exposure_used_yr.append(np.nan)
            assets_redeploy_used_yr.append(np.nan)
            status_list.append("MISSING_TOTAL_ASSETS")
            struct_income_yr.append(np.nan)
            lost_nii_yr.append(np.nan)
            continue

        donor_split = None
        # Highest precedence: explicit overrides passed in by caller
        if donor_split_override_by_bank and bank in donor_split_override_by_bank:
            donor_split = donor_split_override_by_bank[bank]
        else:
            # Next: use per-bank donor split % columns from the CSV (if present),
            # falling back to DEFAULT_DONOR_SPLIT_OF_TOTAL_ASSETS for missing donors.
            if bank in bmap.index:
                donor_split = donor_split_from_row(bmap.loc[bank])

        donor_expo = build_donor_exposures_from_total_assets(total_assets_bn, donor_split)


        
        # --- STOCK ASSUMPTION FOR DONOR ASSETS ---
        # donor_expo is a *stock* of eligible donor assets (EUR bn). In the *flow / steady-state* setup, we treat the donor sub-portfolios as an
        # annual origination flow that replenishes each year.
        # Annual eligible donor flow (EUR bn / yr) = donor stock * rolling fraction (0..25%, default 15%).
        _roll = donor_roll_pct_by_donor or {}
        _donor_expo_flow = {}
        for _d, _v in donor_expo.items():
            try:
                _rp = float(_roll.get(_d, 15.0))
            except Exception:
                _rp = 15.0
            if not np.isfinite(_rp):
                _rp = 15.0
            _rp = max(min(_rp, 25.0), 0.0)
            _donor_expo_flow[_d] = float(_v) * (_rp / 100.0)
        donor_expo = _donor_expo_flow

        # Partner share (investor participation)
        sale_share_dec = max(min(float(sale_share_pct) / 100.0, 0.999999), 0.0)

        retained_risk_dec = max(min(float(retained_risk_pct) / 100.0, 1.0), 0.0)
        net_relief_dec = sale_share_dec * (1.0 - retained_risk_dec)

        # ------------------------------------------------------------
        # Funded risk participation economics (bank-only)
        # ------------------------------------------------------------
        # Structural income rates (decimal, per year):
        #   - skim applies to investor share only (p)
        #   - servicing applies to total portfolio
        #   - upfront fee is annualised over the horizon
        gain_on_sale_dec = float(gain_on_sale_bps) / 10000.0
        servicing_dec = float(servicing_fee_bps) / 10000.0
        origination_fee_dec_ann = (float(origination_fee_bps) / 10000.0)

        struct_rate_dec = origination_fee_dec_ann + (sale_share_dec * (servicing_dec + gain_on_sale_dec))
        struct_rate_after_tax = struct_rate_dec * (1.0 - tx_i)

        # Estimate *CET1-step* donor exposure needed (so we can net structural income from the ROE profit target).
        # We approximate using the exposure-weighted average donor RW across available donor buckets.
        _rw_num = 0.0
        _rw_den = 0.0
        for _d, _expo in donor_expo.items():
            _rw = float(DONOR_RISK_WEIGHT.get(_d, 0.0))
            if _rw > 0 and _expo > 0:
                _rw_num += _expo * _rw
                _rw_den += _expo
        avg_donor_rw = (_rw_num / _rw_den) if _rw_den > 0 else 0.0

        expo_needed_cet1_est = 0.0
        if avg_donor_rw > 0 and net_relief_dec > 0:
            expo_needed_cet1_est = float(rwa_target_cet1_yr) / (avg_donor_rw * net_relief_dec)

        structural_income_cet1_est = expo_needed_cet1_est * struct_rate_after_tax

        # Reduce the redeployment profit target by the estimated structural income from the CET1 step.
        # (This avoids systematically overshooting the ROE target.)
        profit_target_redeploy_yr = max(0.0, float(profit_target_redeploy_yr) - float(structural_income_cet1_est))



        # ------------------------------
        # Step 1 + Step 2: solve ROE target on TOTAL uplift (incl structural income)
        # ------------------------------

        # Helper: run the two-step allocation for a given redeploy profit target
        def _run_once(_profit_target_redeploy_yr: float) -> Dict[str, object]:
            _pt = float(_profit_target_redeploy_yr) if _profit_target_redeploy_yr is not None else 0.0
            if not np.isfinite(_pt) or _pt < 0:
                _pt = 0.0

            # Step 1: Redeployment allocation (profit-target driven)
            _alloc_redeploy = allocate_until_profit_target(
                _pt,
                donor_expo,
                sale_share_dec,
                retained_risk_dec=retained_risk_dec,
                tax_dec=tx_i,
                receiver_split_by_donor=receiver_split_by_donor,
                donor_roll_pct_by_donor=donor_roll_pct_by_donor,
            )

            _achieved_redeploy = float(_alloc_redeploy.get("total_rwa_reduction_eur_bn", 0.0) or 0.0)
            _expo_used_redeploy = float(_alloc_redeploy.get("total_exposure_used_eur_bn", 0.0) or 0.0)
            _assets_redeploy_used = float(_alloc_redeploy.get("total_assets_redeploy_used_eur_bn", 0.0) or 0.0)
            _profit_redeploy = float(_alloc_redeploy.get("total_profit_eur_bn_yr", 0.0) or 0.0)
            _status_redeploy = str(_alloc_redeploy.get("status", "OK"))

            # Net out donor exposure already consumed in Step 1
            _used_by_donor = {}
            for _a in _alloc_redeploy.get("allocations", []):
                _used_by_donor[_a.donor] = _used_by_donor.get(_a.donor, 0.0) + float(_a.exposure_used_eur_bn)

            _donor_expo_remaining = dict(donor_expo)
            for _d_k, _used in _used_by_donor.items():
                if _d_k in _donor_expo_remaining:
                    _donor_expo_remaining[_d_k] = max(0.0, float(_donor_expo_remaining[_d_k]) - float(_used))

            # Step 2: CET1 uplift allocation (capacity after Step 1)
            _alloc_cet1 = allocate_rwa_reduction_equal_receivers(
                (rwa_target_cet1_yr + _achieved_redeploy),
                _donor_expo_remaining,
                sale_share_dec,
                retained_risk_dec=retained_risk_dec,
                receiver_split_by_donor=receiver_split_by_donor,
                donor_roll_pct_by_donor=donor_roll_pct_by_donor,
            )

            _achieved_cet1 = float(_alloc_cet1.get("total_rwa_reduction_eur_bn", 0.0) or 0.0)
            _expo_used_cet1 = float(_alloc_cet1.get("total_exposure_used_eur_bn", 0.0) or 0.0)
            _status_cet1 = str(_alloc_cet1.get("status", "OK"))

            _achieved_total = _achieved_redeploy + _achieved_cet1
            _expo_used_total = _expo_used_redeploy + _expo_used_cet1

            # Structural partnership income applies to TOTAL donor exposure used (REDEPLOY + CET1)
            _struct_income_total = float(_expo_used_total) * float(struct_rate_after_tax)

            # Lost net interest income on SOLD flow (donor net spreads on sold notional).
            # For each donor bucket, sold notional = (eligible flow used) × sale_share.
            _used_by_donor_total: Dict[str, float] = {}
            for _a in list(_alloc_redeploy.get("allocations", [])) + list(_alloc_cet1.get("allocations", [])):
                try:
                    _used_by_donor_total[_a.donor] = _used_by_donor_total.get(_a.donor, 0.0) + float(_a.exposure_used_eur_bn)
                except Exception:
                    pass

            _lost_nii_total = 0.0
            for _d_k, _used_expo in _used_by_donor_total.items():
                _don_sp_bps = float(DONOR_ABS_NET_SPREAD_BPS_BY_B1.get(_d_k, 0.0) or 0.0)
                _lost_nii_total += float(_used_expo) * float(sale_share_dec) * (_don_sp_bps / 10000.0) * (1.0 - tx_i)

            _profit_total = float(_profit_redeploy) + float(_struct_income_total) - float(_lost_nii_total)

            if np.isfinite(cet1_cap_bn) and cet1_cap_bn > 0:
                _roe_achieved_bp = (_profit_total / cet1_cap_bn) * 10000.0
            else:
                _roe_achieved_bp = 0.0

            return {
                "alloc_redeploy": _alloc_redeploy,
                "alloc_cet1": _alloc_cet1,
                "achieved_redeploy": _achieved_redeploy,
                "expo_used_redeploy": _expo_used_redeploy,
                "assets_redeploy_used": _assets_redeploy_used,
                "profit_redeploy": _profit_redeploy,
                "status_redeploy": _status_redeploy,
                "achieved_cet1": _achieved_cet1,
                "expo_used_cet1": _expo_used_cet1,
                "status_cet1": _status_cet1,
                "achieved_total": _achieved_total,
                "expo_used_total": _expo_used_total,
                "struct_income_total": _struct_income_total,
                "lost_nii_total": _lost_nii_total,
                "profit_total": _profit_total,
                "roe_achieved_bp": _roe_achieved_bp,
            }

        # Target TOTAL annual profit implied by the ROE target (EUR bn / yr)
        target_profit_total = (roe_target_bp_i / 10000.0) * float(cet1_cap_bn) if (np.isfinite(cet1_cap_bn) and cet1_cap_bn > 0) else 0.0
        target_profit_total = float(target_profit_total) if np.isfinite(target_profit_total) else 0.0
        target_profit_total = max(target_profit_total, 0.0)

        # Solve for redeploy profit target so that TOTAL ROE uplift (incl structural income) matches roe_target_bp_i
        # We use a robust bracketed bisection because structural income depends on total exposure used (which depends on Step 1).
        _pt_guess = float(profit_target_redeploy_yr)
        _pt_guess = max(_pt_guess, 0.0)

        if not apply_roe_target:
            # Base run: do not solve for an exact ROE target; run once with the direct redeployment profit target.
            _solve = _run_once(_pt_guess)
            profit_target_redeploy_yr = float(_pt_guess)
        else:
            _solve_lo = _run_once(0.0)
            _gap_lo = float(_solve_lo["roe_achieved_bp"]) - float(roe_target_bp_i)

            if _gap_lo >= 0.0:
                # Even with zero redeployment, structural income alone meets/exceeds the target.
                _solve = _solve_lo
                profit_target_redeploy_yr = 0.0
            else:
                # Find an upper bound where we exceed the target
                # Special care when roe_target_bp_i is 0: target_profit_total is 0, and the naive
                # upper bound would be ~0, preventing the bracket from expanding.
                # Use the profit shortfall implied by the zero-redeployment run as a minimum bracket.
                _req_profit = 0.0
                if np.isfinite(cet1_cap_bn) and cet1_cap_bn > 0:
                    try:
                        _req_profit = ((float(roe_target_bp_i) - float(_solve_lo["roe_achieved_bp"])) / 10000.0) * float(cet1_cap_bn)
                    except Exception:
                        _req_profit = 0.0
                _req_profit = max(float(_req_profit), 0.0)
                _hi = max(_pt_guess, target_profit_total, _req_profit, 1e-6)

                _solve_hi = _run_once(_hi)
                _gap_hi = float(_solve_hi["roe_achieved_bp"]) - float(roe_target_bp_i)

                _n_expand = 0
                while _gap_hi < 0.0 and _n_expand < 12:
                    _hi *= 2.0
                    _solve_hi = _run_once(_hi)
                    _gap_hi = float(_solve_hi["roe_achieved_bp"]) - float(roe_target_bp_i)
                    _n_expand += 1

                if _gap_hi < 0.0:
                    # Not feasible: even very high redeployment profit target can't meet the ROE target (capacity constrained)
                    _solve = _solve_hi
                    profit_target_redeploy_yr = float(_hi)
                else:
                    _lo = 0.0
                    _best = _solve_hi
                    _best_pt = _hi

                    # Bisection to within ~0.05 bp (more than enough precision for display)
                    for _ in range(28):
                        _mid = 0.5 * (_lo + _hi)
                        _solve_mid = _run_once(_mid)
                        _gap_mid = float(_solve_mid["roe_achieved_bp"]) - float(roe_target_bp_i)

                        # Track best
                        if abs(_gap_mid) < abs(float(_best["roe_achieved_bp"]) - float(roe_target_bp_i)):
                            _best = _solve_mid
                            _best_pt = _mid

                        if abs(_gap_mid) <= 0.05:
                            _best = _solve_mid
                            _best_pt = _mid
                            break

                        if _gap_mid < 0.0:
                            _lo = _mid
                        else:
                            _hi = _mid

                    _solve = _best
                    profit_target_redeploy_yr = float(_best_pt)

        # Unpack solved run outputs
        alloc_redeploy = _solve["alloc_redeploy"]
        alloc_cet1 = _solve["alloc_cet1"]

        achieved_redeploy = float(_solve["achieved_redeploy"])
        expo_used_redeploy = float(_solve["expo_used_redeploy"])
        assets_redeploy_used = float(_solve["assets_redeploy_used"])
        profit_redeploy = float(_solve["profit_redeploy"])
        status_redeploy = str(_solve["status_redeploy"])

        achieved_cet1 = float(_solve["achieved_cet1"])
        expo_used_cet1 = float(_solve["expo_used_cet1"])
        status_cet1 = str(_solve["status_cet1"])

        achieved_total = float(_solve["achieved_total"])
        expo_used_total = float(_solve["expo_used_total"])

        struct_income_total = float(_solve.get("struct_income_total", 0.0) or 0.0)
        lost_nii_total = float(_solve.get("lost_nii_total", 0.0) or 0.0)

        # Update reporting targets now that Step 1 (profit-driven) RWA volume is known
        rwa_target_scaled_yr_list[-1] = float(rwa_target_cet1_yr + achieved_redeploy)
        rwa_target_scale_list[-1] = float(achieved_redeploy / rwa_target_cet1_yr) if rwa_target_cet1_yr > 0 else 0.0
# ------------------------------
        # Target compliance checks
        # ------------------------------
        if rwa_target_cet1_yr > 0:
            cet1_gap_pct = abs(achieved_cet1 - rwa_target_cet1_yr) / rwa_target_cet1_yr * 100.0 if achieved_cet1 > 0 else 100.0
        else:
            cet1_gap_pct = 0.0

        if np.isfinite(cet1_cap_bn) and cet1_cap_bn > 0:
            structural_income_total_tmp = float(expo_used_total) * float(struct_rate_after_tax)
            roe_achieved_bp = ((profit_redeploy + structural_income_total_tmp) / cet1_cap_bn) * 10000.0
        else:
            roe_achieved_bp = 0.0

        roe_gap_bp = roe_achieved_bp - roe_target_bp_i

        # Compose a status that reflects both targets
        status = "OK"

        # Step 1 (ROE) feasibility
        if apply_roe_target and roe_target_bp_i > 0:
            if status_redeploy != "OK":
                status = f"ROE_{status_redeploy}"
            else:
                if roe_gap_bp < -1e-6:
                    status = "ROE_TARGET_NOT_MET"

        # Step 2 (CET1) feasibility / tolerance
        if rwa_target_cet1_yr > 0 and status_cet1 != "OK":
            status = f"{status}_CET1_{status_cet1}" if status != "OK" else f"CET1_{status_cet1}"

        if (not require_exact_target) and (rwa_target_cet1_yr > 0) and (cet1_gap_pct > target_tolerance_pct):
            status = f"{status}_CET1_OUTSIDE_TOL({cet1_gap_pct:.2f}%)"

        # ------------------------------
        # Profit calculation (Step 1 ONLY)
        # ------------------------------

        profit = 0.0

        # Audit + profit for Redeployment allocations only
        for a in alloc_redeploy["allocations"]:
            contrib = a.exposure_used_eur_bn * a.delta_s_eff_dec * (1.0 - tx_i)
            profit += contrib

            audit_rows.append({
                "Bank": bank,
                "Scenario": row["Scenario"],
                "SRT_Efficiency": row["SRT_Efficiency"],
                "Step": "REDEPLOY",
                "RWA_target_base_EUR_bn_Yr": rwa_target_yr_base,
                "RWA_target_redeploy_EUR_bn_Yr": rwa_target_redeploy_yr,
                "RWA_target_cet1_EUR_bn_Yr": rwa_target_cet1_yr,
                "RWA_target_total_EUR_bn_Yr": rwa_target_total_yr,
                "Donor": a.donor,
                "Receiver": a.receiver,
                "Exposure_used_EUR_bn_Yr": a.exposure_used_eur_bn,
                "RWA_reduction_EUR_bn_Yr": a.rwa_reduction_eur_bn,
                "Delta_RWA_pp": a.delta_rwa_pp,
                "Abs_net_spread_bps": a.abs_net_spread_bps,
                "Delta_s_eff_dec": a.delta_s_eff_dec,
                "Profit_contrib_EUR_bn_Yr": contrib,
                "Status": status,
            })

        # Audit CET1 allocations (no profit impact)
        for a in alloc_cet1["allocations"]:
            audit_rows.append({
                "Bank": bank,
                "Scenario": row["Scenario"],
                "SRT_Efficiency": row["SRT_Efficiency"],
                "Step": "CET1",
                "RWA_target_base_EUR_bn_Yr": rwa_target_yr_base,
                "RWA_target_redeploy_EUR_bn_Yr": rwa_target_redeploy_yr,
                "RWA_target_cet1_EUR_bn_Yr": rwa_target_cet1_yr,
                "RWA_target_total_EUR_bn_Yr": rwa_target_total_yr,
                "Donor": a.donor,
                "Receiver": a.receiver,
                "Exposure_used_EUR_bn_Yr": a.exposure_used_eur_bn,
                "RWA_reduction_EUR_bn_Yr": a.rwa_reduction_eur_bn,
                "Delta_RWA_pp": a.delta_rwa_pp,
                "Abs_net_spread_bps": a.abs_net_spread_bps,
                "Delta_s_eff_dec": a.delta_s_eff_dec,
                "Profit_contrib_EUR_bn_Yr": 0.0,
                "Status": status,
            })

                # Add structural forward-flow economics and net out lost NII on sold flow
        structural_income_total = float(struct_income_total)
        lost_nii_total = float(lost_nii_total)
        profit_total = float(_solve.get("profit_total", (float(profit) + float(structural_income_total) - float(lost_nii_total))) or 0.0)

        addl_profit_yr.append(profit_total)

        struct_income_yr.append(struct_income_total)
        lost_nii_yr.append(lost_nii_total)

        rwa_red_yr_achieved.append(achieved_total)            # total capacity-consuming RWA reduction achieved
        exposure_used_yr.append(expo_used_total)              # total donor exposure used (Step1 + Step2)
        assets_redeploy_used_yr.append(assets_redeploy_used)  # redeployment assets only (Step 1)
        status_list.append(status)

    df["RWA_reduction_achieved_Yr"] = rwa_red_yr_achieved
    df["RWA_target_base_Yr"] = rwa_target_base_yr_list
    df["RWA_target_scaled_Yr"] = rwa_target_scaled_yr_list
    df["RWA_target_scale"] = rwa_target_scale_list
    df["Assets_redeploy_used_Yr"] = assets_redeploy_used_yr  # receiver-side assets redeployed (implied by receiver risk weights)
    df["Addl_profit_Yr"] = addl_profit_yr
    df["Structural_income_Yr"] = struct_income_yr
    df["Lost_NII_Yr"] = lost_nii_yr
    df["Transition_status"] = status_list

    cap = df["CET1_Capital_EUR_bn"].clip(lower=1e-6)
    df["ROE_delta_bp"] = np.round((df["Addl_profit_Yr"] / cap) * 10000.0, 3)

    out = df[[
        "Bank", "Country", "Region", "Reporting Period",
        "Scenario", "SRT_Efficiency",
        "CET1_Capital_EUR_bn",
        "RWA_reduction_achieved_Yr",
        "RWA_target_base_Yr",
        "RWA_target_scaled_Yr",
        "RWA_target_scale",
        "Assets_redeploy_used_Yr",
        "Structural_income_Yr",
        "Lost_NII_Yr",
        "ROE_delta_bp",
        "gross_eff_lever",
        "Transition_status",
    ]].copy()

    out.attrs["allocations_audit_df"] = pd.DataFrame(audit_rows)
    return out


def compute_sri(sim_df: pd.DataFrame, banks_df: pd.DataFrame) -> pd.DataFrame:
    """
    SRI = 100 * (Gross Assets Offloaded / Bank Total Assets)
    Uses 'Total Assets (EUR bn)' from the bank input file as denominator.
    """
    assets_total = banks_df.set_index("Bank Name")["Total Assets (EUR bn)"].to_dict()
    df = sim_df.copy()
    df["Bank_Assets_Total"] = df["Bank"].map(assets_total)
    denom = df["Bank_Assets_Total"].clip(lower=1e-6)

    df["SRI"] = np.round(100.0 * (df["Gross_Assets_Offloaded_EUR_bn_Yr"] / denom), 3)

    return df[[
        "Bank", "Country", "Region", "Reporting Period",
        "Scenario", "SRT_Efficiency",
        "SRI", "Gross_Assets_Offloaded_EUR_bn_Yr", "Bank_Assets_Total"
    ]]


def to_xlsx_bytes(sim_df: pd.DataFrame, roe_df: pd.DataFrame, sri_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        sim_df.to_excel(writer, sheet_name="Simulation", index=False)
        roe_df.to_excel(writer, sheet_name="ROE", index=False)
        sri_df.to_excel(writer, sheet_name="SRI", index=False)

        # Optional: include allocations audit if present
        audit = roe_df.attrs.get("allocations_audit_df")
        if isinstance(audit, pd.DataFrame) and not audit.empty:
            audit.to_excel(writer, sheet_name="ROE_Transitions", index=False)

    return buf.getvalue()


def make_portfolio_row(banks_sel: pd.DataFrame) -> pd.DataFrame:
    df = banks_sel.copy()

    # Fallbacks for sparse columns (per-bank first)
    df["Net Spread (%)"] = df["Net Spread (%)"].fillna(2.5)
    # SRT cost values from the CSV are not used by the model; keep a consistent fallback for display.
    df["SRT Cost (%)"] = df["SRT Cost (%)"].fillna(DEFAULT_SRT_COST_PCT)
    df["Effective Tax Rate (%)"] = df["Effective Tax Rate (%)"].fillna(0.0)

    # CET1 capital per bank (bn EUR)
    df["_cet1_cap_bn"] = (df["CET1 Ratio (%)"] / 100.0) * df["Total RWA (EUR bn)"]

    total_assets = float(df["Total Assets (EUR bn)"].sum())
    total_rwa = float(df["Total RWA (EUR bn)"].sum())
    total_cet1_cap = float(df["_cet1_cap_bn"].sum())

    # Derived portfolio ratios
    cet1_ratio_pct = (total_cet1_cap / total_rwa) * 100.0 if total_rwa > 0 else np.nan
    rwa_density_pct = (total_rwa / total_assets) * 100.0 if total_assets > 0 else np.nan

    # Asset-weighted averages for spread/cost/tax
    w = df["Total Assets (EUR bn)"].to_numpy(dtype=float)
    wsum = np.nansum(w)

    def wavg(x):
        x = np.asarray(x, dtype=float)
        return float(np.nansum(x * w) / wsum) if wsum > 0 else np.nan

    spread_pct = wavg(df["Net Spread (%)"].to_numpy())
    # Display-only (model uses the SRT cost slider).
    srt_cost_pct = wavg(df["SRT Cost (%)"].to_numpy())
    tax_pct = wavg(df["Effective Tax Rate (%)"].to_numpy())

    
    # If donor split % columns exist, aggregate them to a portfolio-level % (asset-weighted).
    donor_pct_fields: Dict[str, float] = {}
    if any(c in df.columns for c in DONOR_SPLIT_COLS_PCT.values()):
        ta = df["Total Assets (EUR bn)"].to_numpy(dtype=float)
        ta_sum = float(np.nansum(ta))
        for donor, col in DONOR_SPLIT_COLS_PCT.items():
            if col not in df.columns:
                continue
            pct = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 100.0
            expo = float(np.nansum(ta * pct))
            donor_pct_fields[col] = (expo / ta_sum * 100.0) if ta_sum > 0 else np.nan

    portfolio = pd.DataFrame([{
        "Bank Name": "PORTFOLIO (Selected banks)",
        "Country": "—",
        "Region": "—",
        "Reporting Period": "—",
        "Total Assets (EUR bn)": total_assets,
        "Total RWA (EUR bn)": total_rwa,
        "CET1 Ratio (%)": cet1_ratio_pct,
        "RWA Density (%)": rwa_density_pct,
        "Net Spread (%)": spread_pct,
        "SRT Cost (%)": srt_cost_pct,
        "Effective Tax Rate (%)": tax_pct,
        **donor_pct_fields
    }])

    return portfolio


# ============================================================
# ---------------- Streamlit App ----------------
# ============================================================
st.set_page_config(page_title="Bank-specific Offload Simulation", layout="wide")


DATA_PATH = "52_banks_full_results.csv"

try:
    banks = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"""
        ❌ Bank input file not found.

        Expected file:
        {DATA_PATH}

        Please make sure the CSV is located in the same folder as app.py.
        """
    )
    st.stop()

required_cols = [
    "Bank Name",
    "Country",
    "Region",
    "Reporting Period",
    "Scenario",
    "Total Assets (EUR bn)",
    "Total RWA (EUR bn)",
    "CET1 Ratio (%)",
    "Effective Tax Rate (%)",
    "Share_SME_Core",
    "Share_MidCorp_Core",
    "Share_TradeFinance_Core",
    "Share_CRE_Core",
    "Share_ConsumerFinance_Core",
]
missing = [c for c in required_cols if c not in banks.columns]
if missing:
    st.error(f"CSV is missing required columns: {missing}")
    st.stop()

# Clean / ensure types
banks = banks.copy()
banks["Bank Name"] = banks["Bank Name"].astype(str)

banks["Country"] = banks["Country"].astype(str)
banks["Region"] = banks["Region"].astype(str)
banks["Reporting Period"] = banks["Reporting Period"].astype(str)
banks["Scenario"] = banks["Scenario"].astype(str)

# Ensure numeric columns are numeric
for _c in ["Total Assets (EUR bn)", "Total RWA (EUR bn)", "CET1 Ratio (%)", "Effective Tax Rate (%)"]:
    banks[_c] = pd.to_numeric(banks[_c], errors="coerce")

# Derive CET1 capital in EUR bn (needed for ROE calculations)
banks["_cet1_cap_bn"] = (banks["CET1 Ratio (%)"] / 100.0) * banks["Total RWA (EUR bn)"]

# Derive RWA density (for max-cap / outline computations); percent points
banks["RWA Density (%)"] = (banks["Total RWA (EUR bn)"] / banks["Total Assets (EUR bn)"]) * 100.0

# Shares are provided as % of total assets; convert to fractions for internal use
for _c in [
    "Share_SME_Core",
    "Share_MidCorp_Core",
    "Share_TradeFinance_Core",
    "Share_CRE_Core",
    "Share_ConsumerFinance_Core",
]:
    banks[_c] = pd.to_numeric(banks[_c], errors="coerce") / 100.0


# Create legacy donor share columns (percent of total assets) for backward-compatible parts of the app.
# These are used by donor tables and some helper functions that expect donor splits as percentage points.
banks["SME_term"] = banks["Share_SME_Core"] * 100.0
banks["MidCorp_nonIG"] = banks["Share_MidCorp_Core"] * 100.0
banks["Trade_finance"] = banks["Share_TradeFinance_Core"] * 100.0
banks["CRE_non_HVCRE"] = banks["Share_CRE_Core"] * 100.0
banks["Consumer_finance"] = banks["Share_ConsumerFinance_Core"] * 100.0

# Optional legacy columns (kept for backward compatibility / display)
if "Net Spread (%)" not in banks.columns:
    banks["Net Spread (%)"] = np.nan
if "SRT Cost (%)" not in banks.columns:
    banks["SRT Cost (%)"] = np.nan


# Build bank list (used by bank toggles)
bank_list = sorted(banks["Bank Name"].dropna().astype(str).unique().tolist())

# Build country list (used by country multi-select)
if 'Country' in banks.columns:
    country_list = sorted(
        banks['Country'].dropna().astype(str).map(str.strip).replace('', pd.NA).dropna().unique().tolist()
    )
else:
    country_list = []

# ---------------- Top controls header (3 columns) ----------------
# Controls previously in the right "sidebar" column are now rendered in a dashboard-style header.
import itertools

# Add subtle vertical separators between the three top-control columns.
# Implementation note: CSS targeting Streamlit's generated DOM can be brittle across versions.
# We therefore insert two narrow "separator" columns between the three control columns.

_SEPARATOR_STYLE = "border-left: 1px solid rgba(49, 51, 63, 0.20); height: 1300px; margin: 0 auto;"

def _draw_vsep():
    # Large height ensures the line spans the full height of the controls area.
    st.markdown(f"<div style='{_SEPARATOR_STYLE}'></div>", unsafe_allow_html=True)

class _RoundRobinControls:
    def __init__(self, cols):
        self._cols = cols
        self._it = itertools.cycle(range(len(cols)))

    def _next_col(self):
        return self._cols[next(self._it)]

    def __getattr__(self, name):
        col = self._next_col()
        attr = getattr(col, name)
        if callable(attr):
            def _wrapped(*args, **kwargs):
                return attr(*args, **kwargs)
            return _wrapped
        return attr

_top_controls_container = st.container()
with _top_controls_container:
    # Use 5 columns (3 control columns + 2 thin separator columns). Keep gaps small so
    # the 3 control columns retain enough width for sliders/toggles.
    _tc1, _sep1, _tc2, _sep2, _tc3 = st.columns([1, 0.02, 1, 0.02, 1], gap="small")
    with _sep1:
        _draw_vsep()
    with _sep2:
        _draw_vsep()

top_controls = _RoundRobinControls([_tc1, _tc2, _tc3])
top_controls.header("Global Controls")

with _tc1:
    st.subheader("Scenario")
    scenario_bps = st.slider(
        "CET1-uplift target (bp p.a.)",
        min_value=0,
        max_value=400,
        value=168,
        step=1,
        help="Annual CET1-uplift target in basis points",
    )

    # Target ROE uplift (bp) — replaces the old Redeployment/CET1 split slider
    roe_target_bp = st.slider(
        "Annual ROE-uplift target (bps)",
        min_value=0,
        max_value=600,
        value=100,
        step=1,
        help="Target annual ROE uplift in basis points (bp). The model first allocates donor capacity to meet this ROE target (via profit-generating redeployment) and then allocates additional capacity to meet the CET1-uplift target. Caution: The fee-income from the assets/ RWAs left \"free\" to meet the CET1-uplift, represents an implicit floor for ROE-uplift.",
        key="roe_target_slider",
    )


with _tc1:
    st.caption("Select one or more countries:")

    # Default: all countries selected (keeps prior behavior: all banks available)
    if "selected_countries" not in st.session_state:
        st.session_state["selected_countries"] = (["Germany"] if "Germany" in country_list else ([country_list[0]] if len(country_list) else []))

    # Track the previous country selection so we can detect changes and sync bank selections.
    if "prev_selected_countries" not in st.session_state:
        st.session_state["prev_selected_countries"] = list(st.session_state.get("selected_countries", []))

    selected_countries = st.multiselect(
        "Countries",
        options=country_list,
        default=st.session_state.get("selected_countries", ["Germany"]),
        key="selected_countries_multiselect",
        help="Filter the bank universe by country",
    )

    selected_countries = list(selected_countries)
    prev_selected_countries = list(st.session_state.get("prev_selected_countries", []))
    countries_changed = set(selected_countries) != set(prev_selected_countries)

    st.session_state["selected_countries"] = selected_countries

    # Apply the country filter to the bank universe
    if selected_countries:
        banks_f = banks[banks["Country"].astype(str).isin(selected_countries)].copy()
    else:
        banks_f = banks.copy()

    bank_list = sorted(banks_f["Bank Name"].dropna().astype(str).unique().tolist())

    # ------------------------------------------------------------
    # Country -> Bank linkage
    # ------------------------------------------------------------
    # If the country selection changed, automatically (de)select banks so that:
    #   - selecting a country selects *all* banks in that country
    #   - deselecting a country deselects *all* banks in that country
    # After the initial sync on country change, users can still fine-tune banks manually.
    if countries_changed:
        if selected_countries:
            auto_banks = list(bank_list)  # all banks in the selected countries
        else:
            auto_banks = []

        st.session_state["selected_banks"] = auto_banks
        # Also set the widget state so the bank multiselect reflects the sync immediately.
        st.session_state["selected_banks_multiselect"] = auto_banks
        st.session_state["prev_selected_countries"] = selected_countries

    st.markdown("---")
    st.caption("Select one or more banks:")

    # Defaults: prefer LBBW and NordLB if present; otherwise fall back to first bank
    preferred_defaults = [b for b in ["LBBW", "NordLB"] if b in bank_list]
    if not preferred_defaults and bank_list:
        preferred_defaults = [bank_list[0]]

    # Use a single multi-select dropdown instead of many checkboxes
    if "selected_banks" not in st.session_state:
        st.session_state["selected_banks"] = preferred_defaults

    # If the country filter removed some previously selected banks, drop them
    prev_banks = st.session_state.get("selected_banks", [])
    prev_banks = [b for b in prev_banks if b in bank_list]
    default_banks = prev_banks if prev_banks else preferred_defaults

    selected_banks = st.multiselect(
        "Banks",
        options=bank_list,
        default=st.session_state.get("selected_banks_multiselect", default_banks),
        key="selected_banks_multiselect",
        help="Select one or more banks",
    )

    # Keep the rest of the app compatible (it expects st.session_state['selected_banks'])
    st.session_state["selected_banks"] = list(selected_banks)


# ------------------------------------------------------------
# Consistent legend + colors across charts (by Bank)
# ------------------------------------------------------------
# Keep a stable order for legend items (use the sidebar order)
BANK_ORDER = [b for b in bank_list if b in selected_banks]

# Use the active Plotly template colorway (matches PX defaults)
BANK_COLOR_SEQ = list(getattr(pio.templates[pio.templates.default].layout, "colorway", []))
if not BANK_COLOR_SEQ:
    BANK_COLOR_SEQ = px.colors.qualitative.Plotly

# Deterministic mapping: Bank -> color, in BANK_ORDER
BANK_COLOR_MAP = {b: BANK_COLOR_SEQ[i % len(BANK_COLOR_SEQ)] for i, b in enumerate(BANK_ORDER)}


override_tax_rate = None
sale_share = 70
gain_on_sale_bps = 50
servicing_fee_bps = 20
origination_fee_bps = 0
retained_risk_pct = 0
receiver_split_by_donor = {}


with _tc2:
    st.subheader("Whole-loan forward flow")

    sale_share = st.slider(
        "Sold share of eligible flow (%)" ,
        min_value=10,
        max_value=95,
        value=70,
        step=1,
        help="Of the annual eligible (rolling) flow, the fraction that is sold to the strategic partner (whole-loan forward flow). The bank retains the remainder.",
        key="sale_share_slider",
    )


    retained_risk_pct = st.slider(
        "Retained risk on sold flow (%)",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
        help="Portion of the credit risk of the SOLD loans that the bank still bears (e.g., first-loss, guarantee, recourse). 0% = clean true sale. This reduces net RWA relief from selling.",
        key="retained_risk_pct_slider",
    )

    gain_on_sale_bps = st.slider(
        "Gain-on-sale (bps, on sold flow)",
        min_value=0,
        max_value=200,
        value=100,
        step=1,
        help="Upfront pricing margin on the sold loans (positive = sold above par; negative = sold below par). Modeled as bps on the sold flow.",
        key="gain_on_sale_bps_slider",
    )

    servicing_fee_bps = st.slider(
        "Servicing fee (bps, on sold flow)",
        min_value=0,
        max_value=100,
        value=20,
        step=1,
        help="Ongoing servicing fee earned by the bank on the loans sold to the partner (partner-owned book). Modeled as bps on the sold flow.",
        key="servicing_fee_bps_slider",
    )

    origination_fee_bps = st.slider(
        "Upfront/origination fee (bps, one-off)",
        min_value=0,
        max_value= 300,
        value= 100,
        step=1,
        help="One-off origination fee earned on all eligible (rolling) flow, whether sold or retained. In steady-state, this is treated as an annual uplift on the eligible rolling flow.",
        key="origination_fee_bps_slider",
    )

    st.markdown("---")
    # Bank-level capacity indicator placeholder (filled later after simulation)
    st.markdown("**Capacity indicator (eligible donor RWAs)**")
    bank_capacity_placeholder = st.container()


    


with _tc3:
    st.subheader("Transition engine controls")

    st.markdown("**Rolling fraction of donor sub-portfolios (annual origination flow)**")
    st.caption("Steady-state flow assumption: each year, this % of each donor sub-portfolio is renewed by new business. The transition engine allocates *annual* flow; sold share applies to that flow.")

    roll_sme = st.slider("Rolling fraction — SME core (% p.a.) - RW 90%", 0, 25, 15, 1, key="roll_sme")
    roll_mid = st.slider("Rolling fraction — Mid-corp core (% p.a.) - RW 70%", 0, 25, 15, 1, key="roll_mid")
    roll_tf  = st.slider("Rolling fraction — Trade finance core (% p.a.) - RW 40%", 0, 25, 15, 1, key="roll_tf")
    roll_cre = st.slider("Rolling fraction — CRE core (% p.a.) - RW 80%", 0, 25, 15, 1, key="roll_cre")
    roll_cons = st.slider("Rolling fraction — Consumer finance core (% p.a.) - RW 75%", 0, 25, 15, 1, key="roll_cons")

    donor_roll_pct = {
        "B1_SME_CORE": float(roll_sme),
        "B1_MIDCORP_CORE": float(roll_mid),
        "B1_TRADE_FIN_CORE": float(roll_tf),
        "B1_CRE_CORE": float(roll_cre),
        "B1_CONSUMER_FIN_CORE": float(roll_cons),
    }


    st.markdown("### Receiver split per donor")
    st.caption(
    "Choose how each donor bucket is split across receiver portfolios. "
    "Within the hard guardrails, a cell is selectable only if the eligibility product (margin / risk-weight rule) is positive. "
    "Each donor row should sum to 100%; rows that do not sum to 100% are treated as invalid (no flow allocated) until corrected."
)

    def _bucket_label(x: str) -> str:
        s = str(x)
        if s in {"B1_SME_CORE"}:
            return "SME core"
        if s in {"B1_MIDCORP_CORE"}:
            return "Mid-corp core"
        if s in {"B1_TRADE_FIN_CORE"}:
            return "Trade finance core"
        if s in {"B1_CRE_CORE"}:
            return "CRE core"
        if s in {"B1_CONSUMER_FIN_CORE"}:
            return "Consumer finance core"

        if s in {"B2_SME_RISK_UP"}:
            return "SME risk-up"
        if s in {"B2_MIDCORP_RISK_UP"}:
            return "Mid-corp risk-up"
        if s in {"B2_STRUCTURED_WC"}:
            return "Structured working capital"
        if s in {"B2_CRE_RISK_UP"}:
            return "CRE risk-up"
        if s in {"B2_SPECIALTY_ABL"}:
            return "Specialty / ABL"
        return s.replace("B1_", "").replace("B2_", "").replace("_", " ").title()

    _donor_labels = {_d: _bucket_label(_d) for _d in B1_DONORS}
    _recv_labels = {_r: _bucket_label(_r) for _r in B2_RECEIVERS}

    # ---------------------------------------------------------------------
    # Hard-coded eligibility matrix for donor -> receiver transitions.
    # Disallowed cells are fixed at 0% and non-editable in the UI.
    # ---------------------------------------------------------------------
    _ALLOWED_RECEIVERS_BY_DONOR: Dict[str, set] = {
        # SME core: allow all receivers (incl. Specialty / ABL)
        "B1_SME_CORE": set(B2_RECEIVERS),
        # Mid-corp core: allow all receivers
        "B1_MIDCORP_CORE": set(B2_RECEIVERS),
        # Trade finance core: allow all receivers (as requested)
        "B1_TRADE_FIN_CORE": set(B2_RECEIVERS),
        # CRE core: allow all receivers (as requested)
        "B1_CRE_CORE": set(B2_RECEIVERS),
        # Consumer finance core: allow all receivers (no hard disablement)
        "B1_CONSUMER_FIN_CORE": set(B2_RECEIVERS),
    }
    
    # Persisted user-controlled section: "Receiver split per donor"
    # - Eligibility is determined by:
    #     (a) hard allow-list (_ALLOWED_RECEIVERS_BY_DONOR) AND
    #     (b) transition_eligibility_product(donor, receiver) > 0 (finite).
    # - Non-eligible cells are greyed out and forced to 0.
    # - Row-sum constraint: for each donor, values across *eligible* receivers are forced to sum to 100%.
    #   Implemented by making the last eligible receiver a computed "remainder to 100%" cell (disabled input).
    
    # Compute eligible receiver sets per donor (intersection of hard-guardrails and eligibility product)
    _ELIGIBLE_BY_DONOR: Dict[str, set] = {}
    for _d in B1_DONORS:
        _elig = set()
        for _r in B2_RECEIVERS:
            if _r not in _ALLOWED_RECEIVERS_BY_DONOR.get(_d, set()):
                continue
            try:
                prod = float(transition_eligibility_product(_d, _r))
            except Exception:
                prod = float("nan")
            if np.isfinite(prod) and prod > 0.0:
                _elig.add(_r)
        _ELIGIBLE_BY_DONOR[_d] = _elig
    
    # Default: equal split across eligible receivers (last eligible receiver will be the remainder)
    if "receiver_split_pivot" not in st.session_state:
        _rows = []
        for _d in B1_DONORS:
            elig = [r for r in B2_RECEIVERS if r in _ELIGIBLE_BY_DONOR.get(_d, set())]
            n_elig = len(elig)
            row = {"Donor": _donor_labels[_d], "_donor_id": _d}
            for _r in B2_RECEIVERS:
                row[_recv_labels[_r]] = 0.0
            if n_elig > 0:
                base = 100.0 / float(n_elig)
                for _r in elig[:-1]:
                    row[_recv_labels[_r]] = base
                row[_recv_labels[elig[-1]]] = 100.0 - base * float(max(n_elig - 1, 0))
            _rows.append(row)
        st.session_state["receiver_split_pivot"] = pd.DataFrame(_rows)
    
    _cur = st.session_state["receiver_split_pivot"].copy()
    
    # Render editable grid (number inputs) and build split dict for the allocator
    receiver_split_by_donor: Dict[str, Dict[str, float]] = {}
    _rows_out = []
    
    _hcols = st.columns([1.4] + [1.0] * len(B2_RECEIVERS), gap="small")
    _hcols[0].markdown("**Donor**")
    for j, _r in enumerate(B2_RECEIVERS, start=1):
        _hcols[j].markdown(f"**{_recv_labels[_r]}**")
    
    for _d in B1_DONORS:
        elig_set = _ELIGIBLE_BY_DONOR.get(_d, set())
        elig = [r for r in B2_RECEIVERS if r in elig_set]
    
        # Choose remainder cell: prefer the last receiver in the global order if eligible; otherwise last eligible.
        remainder_r = (
            (B2_RECEIVERS[-1] if (B2_RECEIVERS and B2_RECEIVERS[-1] in elig_set) else (elig[-1] if len(elig) > 0 else None))
        )
    
        _rcols = st.columns([1.4] + [1.0] * len(B2_RECEIVERS), gap="small")
        _rcols[0].markdown(_donor_labels[_d])
    
        vals_pct: Dict[str, float] = {r: 0.0 for r in B2_RECEIVERS}
        running = 0.0  # running sum across eligible receivers except remainder
    
        for j, _r in enumerate(B2_RECEIVERS, start=1):
            col = _recv_labels[_r]
    
            # Pull previous value if present
            try:
                prev = float(_cur.loc[_cur["_donor_id"] == _d, col].iloc[0])
            except Exception:
                prev = 0.0
    
            # Non-eligible transition: fixed 0 and not editable
            if _r not in elig_set:
                _k = f"receiver_split_{_d}_{_r}"
                st.session_state[_k] = 0.0
                _rcols[j].number_input(
                    label="",
                    min_value=0.0,
                    max_value=0.0,
                    value=0.0,
                    step=1.0,
                    key=_k,
                    disabled=True,
                )
                vals_pct[_r] = 0.0
                continue
    
            # Remainder cell (last eligible receiver): computed and disabled
            if remainder_r is not None and _r == remainder_r:
                rem = 100.0 - running
                if not np.isfinite(rem):
                    rem = 0.0
                rem = float(max(0.0, min(100.0, rem)))
    
                _k = f"receiver_split_{_d}_{_r}"
                st.session_state[_k] = float(rem)
    
                _rcols[j].number_input(
                    label="",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(rem),
                    step=1.0,
                    key=_k,
                    disabled=True,
                    help="Remainder to 100% (computed).",
                )
                vals_pct[_r] = rem
                continue
    
            # Editable eligible cell: cap input so row sum can never exceed 100
            remaining = 100.0 - running
            if not np.isfinite(remaining):
                remaining = 100.0
            remaining = float(max(0.0, min(100.0, remaining)))
    
            v = _rcols[j].number_input(
                label="",
                min_value=0.0,
                max_value=float(remaining),
                value=float(max(0.0, min(prev, remaining))),
                step=1.0,
                key=f"receiver_split_{_d}_{_r}",
            )
            v = float(v or 0.0)
            v = float(max(0.0, min(v, remaining)))
            vals_pct[_r] = v
            running += v
    
        # Build allocator weights: fractions summing to 1 across eligible receivers (if any)
        row_sum = float(sum(vals_pct[r] for r in elig)) if elig else 0.0
        if row_sum > 0:
            receiver_split_by_donor[_d] = {r: (vals_pct[r] / row_sum) for r in elig if vals_pct[r] > 0}
        else:
            receiver_split_by_donor[_d] = {}
    
        # Persist the displayed percentages (including remainder)
        out_row = {"Donor": _donor_labels[_d], "_donor_id": _d}
        for _r in B2_RECEIVERS:
            out_row[_recv_labels[_r]] = float(vals_pct.get(_r, 0.0))
        _rows_out.append(out_row)
    
    st.session_state["receiver_split_pivot"] = pd.DataFrame(_rows_out)

# donor_roll_pct is defined in Transition engine controls (do not overwrite)


# Optional per-bank donor split override (not used unless you add controls for it)
donor_split_override = None

# Sidebar toggles removed as requested
require_exact = False
# Target tolerance (%) slider removed; fixed default used instead.
tol_pct = 0.5
show_audit = False

# Placeholder for capacity indicator (filled after model run)


# Offload Display toggles removed as requested (fixed defaults)
metric = "Assets (EUR bn)"
agg = "Total (Horizont)"


top_controls.markdown("---")
top_controls.markdown("---")
# Single SRT efficiency (replaces A/B/C/D sliders)
top_controls.markdown("---")

# Validate selections
if not selected_banks:
    st.error("Please select at least one bank in the controls above.")
    st.stop()

banks_sel = banks_f[banks_f["Bank Name"].isin(selected_banks)].copy()

# Build single-scenario dict
scenarios = {"Banks": scenario_bps}

effs = [round(float(sale_share) / 100.0, 6)]

portfolio_df = make_portfolio_row(banks_sel)

# Run model for portfolio
sim_port = simulate_offload( portfolio_df, scenarios, effs)
roe_port = compute_roe_delta_transitions_greedy(
    sim_port,
    portfolio_df,
    roe_target_bp=roe_target_bp,
    sale_share_pct=sale_share,
    retained_risk_pct=retained_risk_pct,

    gain_on_sale_bps=gain_on_sale_bps,
    servicing_fee_bps=servicing_fee_bps,
    origination_fee_bps=origination_fee_bps,
    apply_roe_target=True,
    
    override_tax_rate=override_tax_rate,
    require_exact_target=require_exact,
    target_tolerance_pct=tol_pct,
    donor_roll_pct_by_donor=donor_roll_pct,
                        receiver_split_by_donor=receiver_split_by_donor,
)

# Base (unscaled) allocator run for chart (1) base bars
roe_port_base = compute_roe_delta_transitions_greedy(
    sim_port,
    portfolio_df,
    roe_target_bp=roe_target_bp,
    sale_share_pct=sale_share,
    retained_risk_pct=retained_risk_pct,

    gain_on_sale_bps=gain_on_sale_bps,
    servicing_fee_bps=servicing_fee_bps,
    origination_fee_bps=origination_fee_bps,
    apply_roe_target=False,
    
    override_tax_rate=override_tax_rate,
    require_exact_target=require_exact,
    target_tolerance_pct=tol_pct,
    donor_roll_pct_by_donor=donor_roll_pct,
                        receiver_split_by_donor=receiver_split_by_donor,
)
sri_port = compute_sri(sim_port, portfolio_df)

# ---- Run model ----
sim_df = simulate_offload( banks_sel, scenarios, effs)

# --- Safety: guarantee transition-based offload columns exist (downstream reports expect them)
for _c in [
    'Assets_Offloaded_CET1_Transition_EUR_bn_Yr',
    'Assets_Offloaded_ROE_Transition_EUR_bn_Yr',
]:
    if _c not in sim_df.columns:
        sim_df[_c] = 0.0

roe_df = compute_roe_delta_transitions_greedy(
    sim_df,
    banks_sel,
    roe_target_bp=roe_target_bp,
    sale_share_pct=sale_share,
    gain_on_sale_bps=gain_on_sale_bps,
    servicing_fee_bps=servicing_fee_bps,
    origination_fee_bps=origination_fee_bps,
    apply_roe_target=True,
    
    override_tax_rate=override_tax_rate,
    require_exact_target=require_exact,
    target_tolerance_pct=tol_pct,
    donor_roll_pct_by_donor=donor_roll_pct,
                        receiver_split_by_donor=receiver_split_by_donor,
)

# Attach transition-based offload metrics derived from the ROE allocator audit trail

def _attach_transition_based_assets(sim: pd.DataFrame, roe: pd.DataFrame) -> pd.DataFrame:
    """Attach partnership asset offload columns to the simulation output.

    For funded risk participation, the "offloaded assets" are the investor participation
    share (p) of the donor exposure that is put into the participation structure.

    We compute step-wise, using the allocation audit trail:

      - ROE step  ("REDEPLOY")
      - CET1 step ("CET1")

    For each step:
      1) Aggregate donor exposure used (EUR bn / yr):
           X_step = sum Exposure_used_EUR_bn_Yr
      2) Convert to offloaded assets using partner share p:
           Assets_offloaded_step = p * X_step
      3) (no time horizon): Assets_offloaded_step_EUR_bn_Yr is the steady-state annual amount

    Columns attached (EUR bn):
      - Assets_Offloaded_ROE_Transition_EUR_bn_Yr 
      - Assets_Offloaded_CET1_Transition_EUR_bn_Yr 
      - Assets_Offloaded_Transition_EUR_bn_Yr    (sum of both steps)

    We also merge back endogenous scaling outputs from the ROE engine:
      - RWA_target_base_Yr
      - RWA_target_scaled_Yr
      - RWA_target_scale
    """
    sim = sim.copy()

    sim = sim.drop(columns=[
        "Assets_Offloaded_ROE_Transition_EUR_bn_Yr",
        "Assets_Offloaded_ROE_Transition_EUR_bn_Yr",
        "Assets_Offloaded_CET1_Transition_EUR_bn_Yr",
        "Assets_Offloaded_CET1_Transition_EUR_bn_Yr",
        "Assets_Offloaded_Transition_EUR_bn_Yr",
        "Assets_Offloaded_Transition_EUR_bn_Yr",
    ], errors="ignore")

    # ---- Optional alternative assets-offload measure (transition-based) ----

    # Scaling outputs are optional; keep schema stable
    for _c in ["RWA_target_base_Yr", "RWA_target_scaled_Yr", "RWA_target_scale"]:
        if _c not in sim.columns:
            sim[_c] = np.nan

    if roe is None or roe.empty:
        # No ROE / transition audit available -> keep schema stable with zeros
        for _c in [
            "Assets_Offloaded_ROE_Transition_EUR_bn_Yr",
            "Assets_Offloaded_ROE_Transition_EUR_bn_Yr",
            "Assets_Offloaded_CET1_Transition_EUR_bn_Yr",
            "Assets_Offloaded_CET1_Transition_EUR_bn_Yr",
            "Assets_Offloaded_Transition_EUR_bn_Yr",
            "Assets_Offloaded_Transition_EUR_bn_Yr",
        ]:
            if _c not in sim.columns:
                sim[_c] = 0.0
        return sim

    audit = roe.attrs.get("allocations_audit_df")
    if not isinstance(audit, pd.DataFrame) or audit.empty:
        return sim

    key = ["Bank", "Scenario", "SRT_Efficiency"]

    # Partner share from UI (percent -> decimal)
    try:
        p = float(sale_share) / 100.0
    except Exception:
        p = 0.70
    p = max(min(p, 0.999999), 0.0)

    # Aggregate exposure used by step
    grp = (
        audit.groupby(key + ["Step"], dropna=False)["Exposure_used_EUR_bn_Yr"]
        .sum()
        .reset_index()
    )

    grp["Assets_offloaded_step_EUR_bn_Yr"] = p * grp["Exposure_used_EUR_bn_Yr"]

    # Pivot to CET1 vs REDEPLOY
    piv = (
        grp.pivot_table(
            index=key,
            columns="Step",
            values="Assets_offloaded_step_EUR_bn_Yr",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )

    piv["Assets_Offloaded_ROE_Transition_EUR_bn_Yr"] = piv["REDEPLOY"] if "REDEPLOY" in piv.columns else 0.0
    piv["Assets_Offloaded_CET1_Transition_EUR_bn_Yr"] = piv["CET1"] if "CET1" in piv.columns else 0.0

    piv["Assets_Offloaded_Transition_EUR_bn_Yr"] = (
        piv["Assets_Offloaded_ROE_Transition_EUR_bn_Yr"] + piv["Assets_Offloaded_CET1_Transition_EUR_bn_Yr"]
    )

    # Merge scaling outputs (may be missing for base runs)
    scale_cols = ["RWA_target_base_Yr", "RWA_target_scaled_Yr", "RWA_target_scale"]
    for c in scale_cols:
        if c not in roe.columns:
            roe[c] = np.nan

    piv = piv.merge(
        roe[key + scale_cols],
        on=key,
        how="left",
        suffixes=("", "_roe"),
    )

    # Merge back to simulation output
    sim = sim.merge(piv, on=key, how="left")

    # Fill NaNs
    for c in [
        "Assets_Offloaded_ROE_Transition_EUR_bn_Yr",
        "Assets_Offloaded_ROE_Transition_EUR_bn_Yr",
        "Assets_Offloaded_CET1_Transition_EUR_bn_Yr",
        "Assets_Offloaded_CET1_Transition_EUR_bn_Yr",
        "Assets_Offloaded_Transition_EUR_bn_Yr",
        "Assets_Offloaded_Transition_EUR_bn_Yr",
    ]:
        if c in sim.columns:
            sim[c] = sim[c].fillna(0.0)

    return sim

sim_df = _attach_transition_based_assets(sim_df, roe_df)

# Base (unscaled) allocator run for chart (1) base bars
roe_df_base = compute_roe_delta_transitions_greedy(
    sim_df,
    banks_sel,
    roe_target_bp=roe_target_bp,
    sale_share_pct=sale_share,
    gain_on_sale_bps=gain_on_sale_bps,
    servicing_fee_bps=servicing_fee_bps,
    origination_fee_bps=origination_fee_bps,
    apply_roe_target=False,
    
    override_tax_rate=override_tax_rate,
    require_exact_target=require_exact,
    target_tolerance_pct=tol_pct,
    donor_roll_pct_by_donor=donor_roll_pct,
                        receiver_split_by_donor=receiver_split_by_donor,
)
sri_df = compute_sri(sim_df, banks_sel)



# ---------------- Top controls (formerly sidebar) ----------------
# Render controls in a true horizontal "header" layout (3 columns).
import itertools

class _RoundRobinControls:
    def __init__(self, cols):
        self._cols = cols
        self._it = itertools.cycle(range(len(cols)))

    def _next_col(self):
        return self._cols[next(self._it)]

    def __getattr__(self, name):
        col = self._next_col()
        attr = getattr(col, name)
        if callable(attr):
            def _wrapped(*args, **kwargs):
                return attr(*args, **kwargs)
            return _wrapped
        return attr

top_controls_container = st.container()
with top_controls_container:
    _c1, _c2, _c3 = st.columns(3, gap="large")

top_controls = _RoundRobinControls([_c1, _c2, _c3])
st.title("Bank-specific Offload Simulation")

# ---- Load bank input data automatically ----
# ---- Main layout: charts left, controls right ----
left_col = st.container()  # charts area (controls moved to top header)


with left_col:
    top_left, top_right = st.columns(2, gap="large")

    with top_left:
        st.subheader("1) CET1-Uplift (end-state)")

        # CET1 uplift is a level (stock) change. It does not "cumulate" the way ROE does.
        # Filled bars show the end-state CET1 uplift target (bp).
        # Light-grey outline shows remaining headroom up to the max-capacity cap (Max. Reg. Divergence, bp).

        def _compute_cet1_target_df():
            base_cols = [c for c in ["Scenario", "SRT_Efficiency", "Bank"] if c in sim_df.columns]
            base = sim_df[base_cols].drop_duplicates().copy()
            try:
                tgt = float(scenario_bps)
            except Exception:
                tgt = 0.0
            base["CET1_target_bp"] = tgt
            return base[["Scenario", "SRT_Efficiency", "Bank", "CET1_target_bp"]]

        def _compute_max_reg_divergence_map_endstate():
            # In steady-state *flow* mode, donor sliders represent annual turnover.
            # The *maximum maintainable* CET1 uplift is driven by the donor STOCK size (as long as roll_pct > 0),
            # because the structure can be refreshed each year on the rolling portion.
            cap_map = {}

            for _, b in banks_sel.iterrows():
                bank = str(b.get("Bank Name", b.get("Bank", ""))).strip()
                if not bank or bank == "nan":
                    continue

                R = float(b.get("Total RWA (EUR bn)", 0.0) or 0.0)
                cet1_ratio = float(b.get("CET1 Ratio (%)", 0.0) or 0.0) / 100.0
                C = cet1_ratio * R

                ta = float(b.get("Total Assets (EUR bn)", 0.0) or 0.0)

                # Donor stock exposure (EUR bn) per bucket from input splits (optionally overridden)
                split = donor_split_from_row(b, DEFAULT_DONOR_SPLIT_OF_TOTAL_ASSETS)
                if donor_split_override and bank in donor_split_override:
                    for d, v in donor_split_override[bank].items():
                        try:
                            split[d] = float(v)
                        except Exception:
                            pass

                donor_expo = {}
                for donor in ['B1_SME_CORE', 'B1_MIDCORP_CORE', 'B1_TRADE_FIN_CORE', 'B1_CRE_CORE', 'B1_CONSUMER_FIN_CORE']:
                    w = float(split.get(donor, 0.0) or 0.0)
                    expo_stock = ta * w

                    # Rolling fraction represents the *annual* eligible flow (EUR bn / yr).
                    roll_pct = float(donor_roll_pct.get(donor, 15.0) or 0.0) if isinstance(donor_roll_pct, dict) else 15.0
                    roll_pct = max(min(roll_pct, 25.0), 0.0)
                    expo_flow = expo_stock * (roll_pct / 100.0) if roll_pct > 0.0 else 0.0

                    donor_expo[donor] = max(0.0, float(expo_flow))

                sale_share_dec = float(sale_share) / 100.0

                try:
                    alloc_cap = allocate_rwa_reduction_equal_receivers(
                        1e12,
                        donor_expo,
                        sale_share_dec,
                        receiver_split_by_donor=receiver_split_by_donor,
                    )
                    eff_max = float(alloc_cap.get("total_rwa_reduction_eur_bn", 0.0) or 0.0)
                except Exception:
                    eff_max = 0.0

                # Avoid division by zero / negative RWAs
                eff_max = min(float(eff_max), max(R - 1e-9, 0.0))

                if eff_max <= 0 or (R - eff_max) <= 0:
                    cap_map[bank] = np.nan
                    continue

                target_ratio = C / (R - eff_max)
                cap_map[bank] = (target_ratio - cet1_ratio) * 10000.0

            return cap_map

        def _plot_cet1_endstate():
            df = _compute_cet1_target_df()
            if df.empty:
                return go.Figure().update_layout(title="CET1 uplift target – Banks (transition-based)")

            scenarios_order = df["Scenario"].astype(str).dropna().unique().tolist()

            def _num0(v):
                try:
                    v = float(v)
                except Exception:
                    return 0.0
                return float(v) if np.isfinite(v) else 0.0
            cap_map = {str(k).strip(): v for k, v in (_compute_max_reg_divergence_map_endstate() or {}).items()}
            banks_in_df = df["Bank"].astype(str).dropna().unique().tolist()

            fig = go.Figure()
            ordered_banks = [b for b in BANK_ORDER if b in banks_in_df] + [b for b in banks_in_df if b not in BANK_ORDER]
            for _i, bank in enumerate(ordered_banks):
                d = df[df["Bank"].astype(str).str.strip() == str(bank).strip()]
                if d.empty:
                    continue

                cur_map = {str(r["Scenario"]): float(r.get("CET1_target_bp", 0.0) or 0.0) for _, r in d.iterrows()}
                cur_vals = [cur_map.get(str(sc), 0.0) for sc in scenarios_order]

                max_val = float(cap_map.get(str(bank).strip(), np.nan))
                pot_vals = [max(max_val - float(v), 0.0) for v in cur_vals] if np.isfinite(max_val) else [0.0 for _ in cur_vals]

                color = BANK_COLOR_MAP.get(bank)

                fig.add_trace(
                    go.Bar(
                        name=bank,
                        x=scenarios_order,
                        y=cur_vals,
                        marker_color=color,
                        offsetgroup=bank,
                        legendgroup=bank,
                        showlegend=True,                    )
                )

                fig.add_trace(
                    go.Bar(
                        name=f"{bank} (potential)",
                        x=scenarios_order,
                        y=pot_vals,
                        base=cur_vals,
                        marker=dict(
                            color="rgba(0,0,0,0)",
                            line=dict(color="lightgrey", width=2),
                        ),
                        offsetgroup=bank,
                        legendgroup=bank,
                        showlegend=False,
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color="lightgrey", width=2),
                    name="CET1 potential (max-capacity)",
                    showlegend=True,
                )
            )

            fig.update_layout(
                barmode="group",
                title="CET1 uplift target (bp) – Banks (transition-based)",
                legend_title_text="Bank",                legend_orientation="v",
                legend_yanchor="top",
                legend_y=1,
                legend_xanchor="left",
                legend_x=1.02,
            )
            fig.update_yaxes(title_text="ΔCET1 ratio (bp, end-state)")
            fig.update_xaxes(title_text="")
            return fig

        st.plotly_chart(_plot_cet1_endstate(), use_container_width=True)
    with top_right:
        st.subheader("2) ROE-Uplift")

        # Annual view only (steady-state; no time horizon)
        def _plot_roe_annual():
            scenarios_order = roe_df["Scenario"].astype(str).dropna().unique().tolist()
            fig = go.Figure()

            # --- Max-capacity ROE potential (outline) ---
            max_roe_map = compute_max_roe_uplift_map(
                sim_df=sim_df,
                banks_sel=banks_sel,
                sale_share_pct=sale_share,
                retained_risk_pct=retained_risk_pct,
                gain_on_sale_bps=gain_on_sale_bps,
                servicing_fee_bps=servicing_fee_bps,
                origination_fee_bps=origination_fee_bps,
                override_tax_rate=override_tax_rate,
                donor_roll_pct_by_donor=donor_roll_pct,
                receiver_split_by_donor=receiver_split_by_donor,
                tol_pct=tol_pct,
            )

            for _i, bank in enumerate(BANK_ORDER):
                d = roe_df[roe_df["Bank"] == bank]
                if d.empty:
                    continue

                cur_map = {str(r["Scenario"]): float(r.get("ROE_delta_bp", 0.0) or 0.0) for _, r in d.iterrows()}
                cur_vals = [cur_map.get(str(sc), 0.0) for sc in scenarios_order]

                max_val = float(max_roe_map.get(bank, np.nan))
                pot_vals = [max(max_val - float(v), 0.0) for v in cur_vals] if np.isfinite(max_val) else [0.0 for _ in cur_vals]

                color = BANK_COLOR_MAP.get(bank)

                fig.add_trace(
                    go.Bar(
                        name=bank,
                        x=scenarios_order,
                        y=cur_vals,
                        marker_color=color,
                        offsetgroup=bank,
                        legendgroup=bank,
                        showlegend=True,
                    )
                )

                fig.add_trace(
                    go.Bar(
                        name=f"{bank} (potential)",
                        x=scenarios_order,
                        y=pot_vals,
                        base=cur_vals,
                        marker=dict(
                            color="rgba(0,0,0,0)",
                            line=dict(color='lightgrey', width=2),
                        ),
                        offsetgroup=bank,
                        legendgroup=bank,
                        showlegend=False,
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color='lightgrey', width=2),
                    name="ROE potential (max-capacity)",
                    showlegend=True,
                )
            )

            fig.update_layout(
                barmode="group",
                title="ΔROE (bp p.a.) – Banks (transition-based)",
                legend_title_text="Bank",
                legend_orientation="v",
                legend_yanchor="top",
                legend_y=1,
                legend_xanchor="left",
                legend_x=1.02,
            )
            fig.update_yaxes(title_text="ΔROE (bp p.a.)")
            fig.update_xaxes(title_text="")
            return fig

        st.plotly_chart(_plot_roe_annual(), use_container_width=True)

    bottom_left, bottom_right = st.columns(2, gap="large")

    with bottom_left:
        st.subheader("3) Total Asset Offload")

        # Annual view only (steady-state; no time horizon)
        offload_df = sim_df  # contains transition-based offload columns via _attach_transition_based_assets

        # Defensive: ensure required transition-based offload columns exist and are numeric
        _off_cols = [
            "Assets_Offloaded_CET1_Transition_EUR_bn_Yr",
            "Assets_Offloaded_ROE_Transition_EUR_bn_Yr",
        ]
        for _c in _off_cols:
            if _c not in offload_df.columns:
                offload_df[_c] = 0.0
            else:
                offload_df[_c] = pd.to_numeric(offload_df[_c], errors="coerce").fillna(0.0)

        def _num0(x) -> float:
            try:
                v = float(x)
            except Exception:
                return 0.0
            return float(v) if np.isfinite(v) else 0.0

        def _build_offload_fig(df: pd.DataFrame, yv_cet1: str, yv_roe: str, y_label: str, title: str):
            fig = go.Figure()
            scenarios_order = df["Scenario"].astype(str).dropna().unique().tolist()

            for bank in BANK_ORDER:
                d = df[df["Bank"] == bank]
                if d.empty:
                    continue
                cet1_map = {str(r["Scenario"]): _num0(r.get(yv_cet1)) for _, r in d.iterrows()}
                roe_map  = {str(r["Scenario"]): _num0(r.get(yv_roe))  for _, r in d.iterrows()}

                base_vals = [cet1_map.get(str(sc), 0.0) for sc in scenarios_order]
                top_vals  = [roe_map.get(str(sc), 0.0)  for sc in scenarios_order]

                color = BANK_COLOR_MAP.get(bank)

                fig.add_trace(
                    go.Bar(
                        name=bank,
                        x=scenarios_order,
                        y=base_vals,
                        marker_color=color,
                        offsetgroup=bank,
                        legendgroup=bank,
                        showlegend=True,
                    )
                )
                fig.add_trace(
                    go.Bar(
                        name=bank,
                        x=scenarios_order,
                        y=top_vals,
                        marker_color=color,
                        opacity=0.35,
                        offsetgroup=bank,
                        legendgroup=bank,
                        showlegend=False,
                    )
                )

            fig.update_layout(
                barmode="stack",
                title=title,
                legend_title_text="Bank",
                legend_orientation="v",
                legend_yanchor="top",
                legend_y=1,
                legend_xanchor="left",
                legend_x=1.02,
)
            fig.update_yaxes(title_text=y_label)
            fig.update_xaxes(title_text="")
            return fig

        fig1_yr = _build_offload_fig(
            offload_df,
            yv_cet1="Assets_Offloaded_CET1_Transition_EUR_bn_Yr",
            yv_roe="Assets_Offloaded_ROE_Transition_EUR_bn_Yr",
            y_label="Assets offloaded (EUR bn p.a., transition-based)",
            title="Required offload (transition-based assets): CET1 + ROE split (annual)",
        )
        st.plotly_chart(fig1_yr, use_container_width=True)

    with bottom_right:

        st.markdown("### 4) Donor utilization – share of eligible donor flow used")
        st.markdown(
            "<div style='color: #666666; font-size: 0.9rem; margin-top: -6px; margin-bottom: 12px;'>"
            "Donor buckets are prioritized according to best transition efficiency = receiver–donor net spread / receiver-donor risk weight improvement. "
            "Transition efficiency descending from left to right."
            "</div>",
            unsafe_allow_html=True,
        )

        # Split utilization into CET1 vs Redeployment steps (stacked; redeployment semi-transparent)

        # Allocations audit (from ROE transition engine) used to split utilization by step
        alloc_f = None
        try:
            alloc_f = getattr(roe_df, "attrs", {}).get("allocations_audit_df", None)
        except Exception:
            alloc_f = None
        if not isinstance(alloc_f, pd.DataFrame):
            alloc_f = pd.DataFrame()
        # Keep only combinations that are currently in sim_df (defensive against stale audits)
        try:
            key_cols = ["Scenario", "SRT_Efficiency", "Bank"]
            if all(c in alloc_f.columns for c in key_cols) and all(c in sim_df.columns for c in key_cols):
                _keys = sim_df[key_cols].drop_duplicates()
                alloc_f = alloc_f.merge(_keys, on=key_cols, how="inner")
        except Exception:
            pass

        if "Step" in alloc_f.columns:
            used_by_step = (
                alloc_f.groupby(["Bank", "Donor", "Step"], as_index=False)
                .agg({"Exposure_used_EUR_bn_Yr": "sum"})
            )
        else:
            # Backward compatible: treat everything as CET1 (no redeployment split available)
            used_by_step = donor_used_bank.copy()
            used_by_step["Step"] = "CET1"

        # Pivot to get per-step exposure used
        used_piv = (
            used_by_step.pivot_table(
                index=["Bank", "Donor"],
                columns="Step",
                values="Exposure_used_EUR_bn_Yr",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reset_index()
        )

        # Standardize column names
        redeploy_col = "REDEPLOY" if "REDEPLOY" in used_piv.columns else None
        cet1_col = "CET1" if "CET1" in used_piv.columns else None

        used_piv["Exposure_REDEPLOY"] = used_piv[redeploy_col] if redeploy_col else 0.0
        used_piv["Exposure_CET1"] = used_piv[cet1_col] if cet1_col else 0.0
        used_piv["Exposure_TOTAL"] = used_piv["Exposure_REDEPLOY"] + used_piv["Exposure_CET1"]

        # Eligible donor exposure stock per bank+donor (EUR bn) after applying donor split + availability cap sliders
        donor_elig = donor_eligible_exposure_long(
            banks_sel,
            donor_roll_pct_by_donor=donor_roll_pct,
            donor_split_override_by_bank=donor_split_override,
        )

        donor_util = used_piv.merge(donor_elig, on=["Bank", "Donor"], how="left")
        donor_util["Eligible_Exposure_EUR_bn"] = donor_util["Eligible_Exposure_EUR_bn"].fillna(0.0)

        # --- STOCK CONSISTENCY ---
        # With donor assets treated as a *stock* (distributed across the horizon inside the annual transition engine),
        # Exposure_CET1 / Exposure_REDEPLOY are per-year usage. For utilization and the capacity indicator we want
        # TOTAL usage over the horizon vs the eligible STOCK pool.
        denom = donor_util["Eligible_Exposure_EUR_bn"].replace(0.0, np.nan)

        donor_util["Util_CET1_pct"] = 100.0 * donor_util["Exposure_CET1"] / denom
        donor_util["Util_REDEPLOY_pct"] = 100.0 * donor_util["Exposure_REDEPLOY"] / denom
        donor_util = donor_util.fillna(0.0)

        donor_util["Util_CET1_pct"] = donor_util["Util_CET1_pct"].clip(lower=0, upper=200)
        donor_util["Util_REDEPLOY_pct"] = donor_util["Util_REDEPLOY_pct"].clip(lower=0, upper=200)

        # Sort for stable display
        donor_util = donor_util.sort_values(["Donor", "Bank"])


        # --- Fill top-control capacity indicators (per selected bank) ---
        try:
            donors_required = ['B1_SME_CORE', 'B1_MIDCORP_CORE', 'B1_TRADE_FIN_CORE', 'B1_CRE_CORE', 'B1_CONSUMER_FIN_CORE']
            # Compute TOTAL utilization (Redeploy + CET1) on a per-bank, per-donor basis
            donor_util["Util_TOTAL_pct"] = (donor_util["Util_CET1_pct"] + donor_util["Util_REDEPLOY_pct"]).clip(lower=0, upper=200)

            bank_full = {}
            for bank in banks_sel["Bank Name"].astype(str).tolist():
                d = donor_util[donor_util["Bank"] == bank]
                if d.empty:
                    bank_full[bank] = False
                    continue
                # Ensure all required donors are present; missing donors => not "full"
                full_flags = []
                for donor in donors_required:
                    row = d[d["Donor"].astype(str) == donor]
                    if row.empty:
                        full_flags.append(False)
                    else:
                        # Treat as full if utilization is ~100% or above
                        full_flags.append(float(row["Util_TOTAL_pct"].iloc[0]) >= 99.9)
                bank_full[bank] = all(full_flags)

            if "bank_capacity_placeholder" in globals():

                with bank_capacity_placeholder:
                    # Preserve the order of banks as selected in the UI
                    sel_order = st.session_state.get("selected_banks", [])
                    # Fallback to banks_sel order if session state is unavailable
                    banks_in_order = sel_order if sel_order else banks_sel["Bank Name"].astype(str).tolist()

                    for bank in banks_in_order:
                        if bank_full.get(bank, False):
                            st.error(f"🚨 {bank}: all eligible donor buckets fully utilized")
                        else:
                            st.success(f"✅ {bank}: capacity available")
        except Exception:
            # Never fail the app due to indicator rendering
            pass

        # Build stacked grouped bars: base=CET1 (opaque), top=REDEPLOY (semi-transparent)
        fig_util = go.Figure()

        donors_order = ['B1_SME_CORE', 'B1_MIDCORP_CORE', 'B1_TRADE_FIN_CORE', 'B1_CRE_CORE', 'B1_CONSUMER_FIN_CORE']

        for bank in BANK_ORDER:
            d = donor_util[donor_util["Bank"] == bank]
            if d.empty:
                continue

            base_map = {str(r["Donor"]): float(r.get("Util_CET1_pct", 0.0) or 0.0) for _, r in d.iterrows()}
            top_map = {str(r["Donor"]): float(r.get("Util_REDEPLOY_pct", 0.0) or 0.0) for _, r in d.iterrows()}

            base_vals = [base_map.get(str(x), 0.0) for x in donors_order]
            top_vals = [top_map.get(str(x), 0.0) for x in donors_order]

            color = BANK_COLOR_MAP.get(bank)

            fig_util.add_trace(
                go.Bar(
                    name=bank,
                    x=donors_order,
                    y=base_vals,
                    marker_color=color,
                    offsetgroup=bank,
                    legendgroup=bank,
                    showlegend=True,
                )
            )

            fig_util.add_trace(
                go.Bar(
                    name=f"{bank} (redeployment)",
                    x=donors_order,
                    y=top_vals,
                    marker_color=color,
                    marker_opacity=0.35,
                    offsetgroup=bank,
                    legendgroup=bank,
                    showlegend=False,
                )
            )

        fig_util.update_layout(
            barmode="stack",
            legend_title_text="Bank",
            legend_orientation="v",
            legend_yanchor="top",
            legend_y=1,
            legend_xanchor="left",
            legend_x=1.02,
        )
        fig_util.update_yaxes(title_text="Utilization (% of eligible per year)")
        fig_util.update_xaxes(title_text="Donor bucket")

        st.plotly_chart(fig_util, use_container_width=True)

# --- Donor bucket weights per selected bank (from CSV) ---
    donor_cols = {
        "SME core (%)": "SME_term",
        "Mid-corp core (%)": "MidCorp_nonIG",
        "Trade finance core (%)": "Trade_finance",
        "CRE core (%)": "CRE_non_HVCRE",
        "Consumer finance core (%)": "Consumer_finance",
    }

    cols_present = [c for c in donor_cols.values() if c in banks_sel.columns]

    if cols_present:
        donor_tbl = (
            banks_sel[["Bank Name"] + cols_present]
            .rename(columns={"Bank Name": "Bank", **{v: k for k, v in donor_cols.items()}})
            .sort_values("Bank")
        )

        
        # Compute per-bank maximum CET1 uplift (US advantage bps) at util=0 that would fully consume donor capacity.
        # In steady-state *flow* mode, max CET1 capacity is driven by donor STOCK size (as long as roll_pct > 0).
        # roll_pct == 0 => no replenishment => no sustainable capacity from that donor bucket.
        try:
            max_adv_map = {}

            for _, b in banks_sel.iterrows():
                bank = str(b.get("Bank Name", b.get("Bank", ""))).strip()
                if not bank or bank == "nan":
                    continue

                R = float(b.get("Total RWA (EUR bn)", 0.0) or 0.0)
                cet1_ratio = float(b.get("CET1 Ratio (%)", 0.0) or 0.0) / 100.0
                C = cet1_ratio * R
                ta = float(b.get("Total Assets (EUR bn)", 0.0) or 0.0)

                split = donor_split_from_row(b, DEFAULT_DONOR_SPLIT_OF_TOTAL_ASSETS)
                if donor_split_override and bank in donor_split_override:
                    for d, v in donor_split_override[bank].items():
                        try:
                            split[d] = float(v)
                        except Exception:
                            pass

                donor_expo = {}
                for donor in ['B1_SME_CORE', 'B1_MIDCORP_CORE', 'B1_TRADE_FIN_CORE', 'B1_CRE_CORE', 'B1_CONSUMER_FIN_CORE']:
                    w = float(split.get(donor, 0.0) or 0.0)
                    expo_stock = ta * w
                    expo_flow = expo_stock * (roll_pct / 100.0)

                    roll_pct = float(donor_roll_pct.get(donor, 15.0) or 0.0) if isinstance(donor_roll_pct, dict) else 15.0
                    roll_pct = max(min(roll_pct, 25.0), 0.0)
                    if roll_pct <= 0.0:
                        expo_stock = 0.0

                    donor_expo[donor] = max(0.0, float(expo_stock))

                sale_share_dec = float(sale_share) / 100.0

                alloc_cap = allocate_rwa_reduction_equal_receivers(
                    1e12,
                    donor_expo,
                    sale_share_dec,
                    receiver_split_by_donor=receiver_split_by_donor,
                )
                eff_max = float(alloc_cap.get("total_rwa_reduction_eur_bn", 0.0) or 0.0)

                # Avoid division by zero / negative RWAs
                eff_max = min(eff_max, max(R - 1e-9, 0.0))

                if eff_max <= 0 or (R - eff_max) <= 0:
                    max_adv_map[bank] = np.nan
                    continue

                target_ratio = C / (R - eff_max)
                max_adv_map[bank] = (target_ratio - cet1_ratio) * 10000.0

            donor_tbl["Max. Reg. Divergence (bp)"] = donor_tbl["Bank"].map(max_adv_map).round(0).astype("Int64")
        except Exception:
            donor_tbl["Max. Reg. Divergence (bp)"] = np.nan

        # --- RWA offload (CET1 uplift + ROE uplift) per bank ---
        try:
            audit_df = getattr(roe_df, "attrs", {}).get("allocations_audit_df", None)
            if isinstance(audit_df, pd.DataFrame) and not audit_df.empty:
                # Sum annual RWA reductions by step and convert to total-horizon RWAs
                by_step = (
                    audit_df.groupby(["Bank", "Step"], dropna=False)["RWA_reduction_EUR_bn_Yr"]
                    .sum()
                    .unstack(fill_value=0.0)
                )
                rwa_cet1_tot = by_step.get("CET1", 0.0) 
                rwa_roe_tot = by_step.get("REDEPLOY", 0.0) 
                rwa_sum_map = (rwa_cet1_tot + rwa_roe_tot).to_dict()
                donor_tbl["RWA freed (EUR bn)"] = donor_tbl["Bank"].map(rwa_sum_map).round(0).astype("Int64").astype("Int64")
            else:
                donor_tbl["RWA freed (EUR bn)"] = np.nan
        except Exception:
            donor_tbl["RWA freed (EUR bn)"] = np.nan

        st.markdown("**Donor bucket weights per selected bank (from input data)**")
        donor_tbl_display = donor_tbl.drop(columns=["Max. Reg. Divergence (bp)"], errors="ignore")

        st.dataframe(
            donor_tbl_display,
            column_config={
                "RWA freed (EUR bn)": st.column_config.NumberColumn(
                    "RWA freed (EUR bn)", format="%.0f"
                )
            },
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.subheader("PORTFOLIO (aggregate across selected banks)")

    # Required offload portfolio = sum of chart (1) across banks (keep CET1/ROE split)
    yl = "Offloaded assets (EUR bn, total, transition-based) — sum across banks"

    _cols_port = [
        "Scenario",
        "SRT_Efficiency",
        "Bank",
        "Assets_Offloaded_CET1_Transition_EUR_bn_Yr",
        "Assets_Offloaded_ROE_Transition_EUR_bn_Yr",
    ]
    # Robust to older / partial sim_df schemas: missing transition-based columns are created as NaN then filled with 0.0
    _tmp_port = sim_df.reindex(columns=_cols_port).copy()
    for _c in ["Assets_Offloaded_CET1_Transition_EUR_bn_Yr", "Assets_Offloaded_ROE_Transition_EUR_bn_Yr"]:
        _tmp_port[_c] = pd.to_numeric(_tmp_port[_c], errors="coerce").fillna(0.0)

    port_base = (
        _tmp_port.groupby(["Scenario", "SRT_Efficiency"], as_index=False)[
            ["Assets_Offloaded_CET1_Transition_EUR_bn_Yr", "Assets_Offloaded_ROE_Transition_EUR_bn_Yr"]
        ].sum()
    )

    port_long = port_base.melt(
        id_vars=["Scenario", "SRT_Efficiency"],
        value_vars=["Assets_Offloaded_CET1_Transition_EUR_bn_Yr", "Assets_Offloaded_ROE_Transition_EUR_bn_Yr"],
        var_name="Component",
        value_name="Assets_offloaded_EUR_bn",
    )
    port_long["Component"] = port_long["Component"].map({
        "Assets_Offloaded_CET1_Transition_EUR_bn_Yr": "CET1-uplift",
        "Assets_Offloaded_ROE_Transition_EUR_bn_Yr": "ROE-uplift",
    })
    # Ensure one bar per Scenario/Component/(SRT_Efficiency) so Plotly can stack reliably
    port_long = (
        port_long.groupby(["Scenario", "SRT_Efficiency", "Component"], as_index=False)["Assets_offloaded_EUR_bn"]
        .sum()
    )


    figP1 = px.bar(
        port_long,
        x="Scenario",
        y="Assets_offloaded_EUR_bn",
        color="Component",
        barmode="stack",
        category_orders={"Component": ["CET1-uplift", "ROE-uplift"]},
        facet_col="SRT_Efficiency" if len(effs) > 1 else None,
        labels={
            "Assets_offloaded_EUR_bn": yl,
            "Scenario": "",
            "SRT_Efficiency": "Partner share",
            "Component": "",
        },
        title="Required Offload – Portfolio (funded risk participation)",
    )

    # Force true stacking (Plotly stacks only within the same offsetgroup; px.bar may set different offsetgroups)
    for _tr in figP1.data:
        _tr.offsetgroup = "portfolio"
        _tr.alignmentgroup = "portfolio"
    figP1.update_layout(barmode="stack")

    # Portfolio ΔROE: weighted average of bank-level ΔROE (from chart 2),
    # weights = Total RWA (EUR bn) × CET1 Ratio (%), taken from input data.
    _w = banks_sel[["Bank Name", "Total RWA (EUR bn)", "CET1 Ratio (%)"]].copy()
    _w["weight"] = _w["Total RWA (EUR bn)"] * _w["CET1 Ratio (%)"]

    _tmp = roe_df.merge(_w[["Bank Name", "weight"]], left_on="Bank", right_on="Bank Name", how="left")
    _tmp = _tmp.dropna(subset=["weight"])

    roe_port = (
        _tmp.groupby(["Scenario", "SRT_Efficiency"], as_index=False)
            .apply(lambda g: pd.Series({"ROE_delta_bp": float(np.average(g["ROE_delta_bp"], weights=g["weight"]))}))
    )

    figP2 = px.bar(
        roe_port,
        x="Scenario",
        y="ROE_delta_bp",
        color="SRT_Efficiency",
        barmode="group",
        labels={"ROE_delta_bp": "ΔROE (bp p.a.)", "Scenario": "", "SRT_Efficiency": "Partner share"},
        title="ΔROE (bp p.a.) – Portfolio (partnership)"
    )
    figP1.update_layout(legend_traceorder="reversed")

    pcol1, pcol2 = st.columns(2, gap="large")
    with pcol1:
        st.plotly_chart(figP1, use_container_width=True)
    with pcol2:
        st.plotly_chart(figP2, use_container_width=True)


# ---- XLSX Export ----
xlsx_bytes = to_xlsx_bytes(sim_df, roe_df, sri_df)
st.download_button(
    label="Donwload XLSX",
    data=xlsx_bytes,
    file_name=f"offload_banks_{pd.Timestamp.today().date()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
