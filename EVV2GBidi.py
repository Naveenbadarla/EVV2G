import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import base64



# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="E.ON EV Charging Optimisation",
    page_icon="⚡",
    layout="wide",
)

# =============================================================================
# GLOBAL PREMIUM E.ON STYLING (works for light & dark themes)
# =============================================================================
st.markdown(
    """
    <style>
    /* Overall background & container */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
    }

    .block-container {
        max-width: 1300px;
        padding-top: 0.5rem;
        padding-bottom: 3rem;
    }

    /* Cards */
    .eon-card {
        background-color: #111827;
        border-radius: 18px;
        padding: 18px 22px;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(148, 163, 184, 0.35);
        color: #e5e7eb;
    }

    .eon-card h3 {
        margin-top: 0;
        color: #f9fafb;
    }

    .eon-card p, .eon-card li {
        color: #e5e7eb;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #111827 !important;
        color: #e5e7eb !important;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.5);
        padding: 10px 12px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #020617 !important;
        border-right: 1px solid rgba(15, 23, 42, 0.9);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f9fafb;
    }

    /* Inputs */
    .stNumberInput input, .stTextInput input {
        border-radius: 10px !important;
    }

    .stFileUploader > label {
        font-weight: 500;
        color: #e5e7eb !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# LOGO LOADING
# =============================================================================
def load_logo(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        return ""

logo_base64 = load_logo("eon_logo.png")

# =============================================================================
# PREMIUM MINIMAL E.ON HEADER
# =============================================================================
st.markdown(
    """
    <style>
    @keyframes subtleShift {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }

    .eon-header {
        width: 100%;
        padding: 24px 40px;
        border-radius: 20px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        justify-content: flex-start;

        background: linear-gradient(90deg,
            #E2000F 0%,
            #D9001A 40%,
            #C90024 100%
        );
        background-size: 220% 220%;
        animation: subtleShift 14s ease-in-out infinite;
        box-shadow: 0 25px 60px rgba(0,0,0,0.5);
    }

    .eon-header-logo {
        width: 170px;
        margin-right: 40px;
        filter: drop-shadow(0 4px 12px rgba(0,0,0,0.4));
    }

    .eon-header-title {
        font-size: 36px;
        font-weight: 600;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.4px;
    }

    .eon-header-sub {
        font-size: 18px;
        font-weight: 300;
        color: rgba(255,255,255,0.9);
        margin-top: 6px;
    }

    .eon-nav {
        margin-top: 12px;
        display: flex;
        gap: 26px;
    }

    .eon-nav a {
        font-size: 15px;
        color: rgba(255,255,255,0.92);
        text-decoration: none;
        transition: 0.25s;
        padding-bottom: 4px;
        border-bottom: 2px solid transparent;
    }

    .eon-nav a:hover {
        border-bottom: 2px solid rgba(255,255,255,0.9);
    }

    @media (max-width: 768px) {
        .eon-header {
            flex-direction: column;
            padding: 22px 20px;
            text-align: center;
        }
        .eon-header-logo {
            margin-right: 0;
            margin-bottom: 12px;
            width: 150px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

header_logo_html = (
    f'<img src="data:image/png;base64,{logo_base64}" class="eon-header-logo">'
    if logo_base64
    else ""
)

st.markdown(
    f"""
    <div class="eon-header">
        {header_logo_html}
        <div>
            <div class="eon-header-title">EV Charging Optimisation Dashboard</div>
            <div class="eon-header-sub">
                Full-Year Day-Ahead & Intraday Market-Based Smart Charging
            </div>
            <div class="eon-nav">
                <a href="#overview">Overview</a>
                <a href="#prices">Market Prices</a>
                <a href="#results">Results</a>
                <a href="#details">Details</a>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<h1 style='text-align:center; font-family:Trebuchet MS, Helvetica, sans-serif; font-weight:800; color:#E2000F; text-shadow:0px 0px 8px rgba(255,255,255,0.15); letter-spacing:1.5px; margin-top:14px;'>E.ON AI<span style='color:white;'>X</span> — AI e<span style='color:white;'>X</span>cellence Initiative</h1>", unsafe_allow_html=True)
st.markdown("""
<style>
@keyframes spin { 
  0% { transform: rotate(0deg); } 
  100% { transform: rotate(360deg); } 
}
.wheel {
  display:inline-block;
  animation: spin 2s linear infinite;
  font-size:22px;
}
</style>

<div style='text-align:center; font-family:Segoe UI, Helvetica, sans-serif; font-size:20px; color:#7A7A7A; margin-top:-8px;'>
Where <span class='wheel'>🛞</span>ptimization meets <span class='wheel'>🛞</span>pportunity — powered by AI.
</div>
""", unsafe_allow_html=True)
from streamlit.components.v1 import html

html(
    """
<style>

.ev-wrapper {
    width: 100%;
    display: flex;
    justify-content: center;
    margin-top: 35px;
    margin-bottom: 40px;
}

.ev-card {
    background: linear-gradient(145deg, #1e1f22, #25262b);
    border-radius: 24px;
    padding: 40px 30px;
    width: 95%;
    max-width: 900px;
    box-shadow: 0 10px 45px rgba(0,0,0,0.7);
    border: 1px solid rgba(255,255,255,0.10);
}

.ev-car {
    width: 240px;
    height: 110px;
    border-radius: 28px;
    background: linear-gradient(180deg, #3a3d42, #2d2f34 80%);
    position: relative;
    margin: 0 auto;
    box-shadow: 0 0 30px rgba(255,0,0,0.25);
}

.ev-wheel {
    width: 46px;
    height: 46px;
    background: #000;
    border-radius: 50%;
    border: 5px solid #8b8b8b;
    position: absolute;
    bottom: -20px;
}
.ev-wheel.left { left: 32px; }
.ev-wheel.right { right: 32px; }

.ev-port {
    width: 16px;
    height: 16px;
    background: #E2000F;
    border-radius: 50%;
    position: absolute;
    right: -12px;
    top: 42px;
    box-shadow: 0 0 14px rgba(226,0,15,1);
}

.ev-cable {
    width: 140px;
    height: 4px;
    background: rgba(255,255,255,0.25);
    position: absolute;
    right: -140px;
    top: 50px;
    border-radius: 2px;
}

.ev-dot {
    width: 10px;
    height: 10px;
    background: #ff3640;
    border-radius: 50%;
    position: absolute;
    animation: cablePulse 1.8s infinite linear;
    box-shadow: 0 0 12px rgba(255,40,40,0.9);
}

.ev-charger {
    width: 90px;
    height: 155px;
    background: linear-gradient(180deg, #2e3036, #1a1a1c 85%);
    border-radius: 16px;
    position: absolute;
    right: -250px;
    top: -12px;
    border: 1px solid rgba(255,255,255,0.15);
}

.ev-charger-screen {
    width: 50px;
    height: 28px;
    background: #E2000F;
    border-radius: 6px;
    margin: 18px auto;
}

@keyframes cablePulse {
    0% { left: 0px; opacity: 1; }
    100% { left: 130px; opacity: 0; }
}

</style>

<div class="ev-wrapper">
    <div class="ev-card">
        <div style="position: relative; height: 180px;">

            <div class="ev-car">
                <div class="ev-wheel left"></div>
                <div class="ev-wheel right"></div>
                <div class="ev-port"></div>
                <div class="ev-cable"></div>

                <div class="ev-dot" style="animation-delay: 0s;"></div>
                <div class="ev-dot" style="animation-delay: 0.35s;"></div>
                <div class="ev-dot" style="animation-delay: 0.7s;"></div>

                <div class="ev-charger">
                    <div class="ev-charger-screen"></div>
                </div>
            </div>

        </div>
        <h3 style='text-align:center;margin-top:22px;color:white;font-weight:300;'>
            Smart Charging… Optimising Your Energy Costs
        </h3>
    </div>
</div>
""",
    height=350,
)

# =============================================================================
# SYNTHETIC PRICE GENERATION (fallback if no upload)
# =============================================================================
def get_synthetic_daily_price_profiles():
    hours = np.arange(24)
    base = 70
    morning_peak = 15 * np.sin((hours - 7) / 24 * 2 * np.pi) ** 2
    evening_peak = 35 * np.sin((hours - 18) / 24 * 2 * np.pi) ** 4
    da_hourly = base + morning_peak + evening_peak  # €/MWh

    rng = np.random.default_rng(42)
    da_q = np.repeat(da_hourly, 4)
    id_q = da_q + rng.normal(0, 6, size=96)

    return da_hourly, id_q

# =============================================================================
# CHARGING FREQUENCY RULES
# =============================================================================
DAY_NAME_TO_IDX = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

def compute_charging_days(pattern, num_days, custom=None, weekly=None):
    if custom is None:
        custom = []

    if pattern == "Every day":
        return list(range(num_days))

    if pattern == "Every other day":
        return list(range(0, num_days, 2))

    if pattern == "Weekdays only (Mon–Fri)":
        return [d for d in range(num_days) if (d % 7) < 5]

    if pattern == "Weekends only (Sat–Sun)":
        return [d for d in range(num_days) if (d % 7) >= 5]

    if pattern == "Custom weekdays":
        allowed = {DAY_NAME_TO_IDX[d] for d in custom}
        return [d for d in range(num_days) if (d % 7) in allowed]

    if pattern == "Custom: X sessions per week":
        weekly = int(weekly)
        out = []
        weeks = num_days // 7
        for w in range(weeks):
            for s in range(weekly):
                day = w * 7 + int(np.floor(s * 7 / weekly))
                if day < num_days:
                    out.append(day)
        return out

    return list(range(num_days))

# =============================================================================
# TIME WINDOW PARSER (15-min resolution)
# =============================================================================
def build_available_quarters(arrival, departure):
    aq = int(arrival * 4) % 96
    dq = int(departure * 4) % 96
    if aq == dq:  # full day
        return list(range(96)), aq, dq
    if aq < dq:
        return list(range(aq, dq)), aq, dq
    return list(range(aq, 96)) + list(range(0, dq)), aq, dq

# =============================================================================
# COSTING LOGIC
# =============================================================================
def apply_tariffs(price_kwh, grid_fee, taxes, vat):
    """
    price_kwh: energy wholesale price in €/kWh
    grid_fee:  grid network charges in €/kWh
    taxes:     taxes & levies in €/kWh
    vat:       VAT in %
    """
    return (price_kwh + grid_fee + taxes) * (1 + vat / 100)

def compute_da_indexed_baseline_daily(E, Pmax, quarters, da_day, grid, taxes, vat):
    """DA-indexed, chronological charging, constant grid fee (no Mod3)."""
    da_q = np.repeat(da_day / 1000.0, 4)  # €/kWh
    remain = E
    cost = 0.0
    emax = Pmax * 0.25  # kWh per quarter

    for q in quarters:
        if remain <= 1e-6:
            break
        e = min(emax, remain)
        cost += e * apply_tariffs(da_q[q], grid, taxes, vat)
        remain -= e

    if remain > 1e-6:
        raise ValueError("Charging window too short to deliver requested energy.")
    return cost

def compute_optimised_daily_cost(E, Pmax, quarters, price_q, grid, taxes, vat):
    """Optimised on wholesale prices only, constant grid fee."""
    pq = price_q / 1000.0
    emax = Pmax * 0.25
    if len(quarters) * emax < E - 1e-6:
        raise ValueError("Charging window too short for requested energy.")

    sorted_q = sorted(quarters, key=lambda x: pq[x])
    remain = E
    cost = 0.0
    for q in sorted_q:
        if remain <= 1e-6:
            break
        e = min(emax, remain)
        cost += e * apply_tariffs(pq[q], grid, taxes, vat)
        remain -= e
    return cost

# =============================================================================
# MODUL 3: HT/ST/NT GRID FEE STRUCTURE
# =============================================================================

# DSO-specific prices (ct/kWh)
dso_tariffs = {
    "Westnetz": {"HT": 15.65, "ST": 9.53, "NT": 0.95},
    "Avacon": {"HT": 8.41, "ST": 6.04, "NT": 0.60},
    "MVV Netze": {"HT": 5.96, "ST": 4.32, "NT": 1.73},
    "MITNETZ": {"HT": 12.60, "ST": 6.31, "NT": 0.69},
    "Stadtwerke München": {"HT": 7.14, "ST": 6.47, "NT": 2.59},
    "Thüringer Energienetze": {"HT": 8.62, "ST": 5.56, "NT": 1.67},
    "LEW": {"HT": 8.09, "ST": 7.09, "NT": 4.01},
    "NetzeBW": {"HT": 13.20, "ST": 7.57, "NT": 3.03},
    "Bayernwerk": {"HT": 9.03, "ST": 4.72, "NT": 0.47},
    "EAM Netz": {"HT": 10.52, "ST": 5.48, "NT": 1.64},
}

# Your DSO-specific quarter validity (True = Modul 3 active)
dso_quarter_valid = {
    "Westnetz":               {1: True, 2: True, 3: True, 4: True},
    "Avacon":                 {1: True, 2: False, 3: False, 4: True},
    "MVV Netze":              {1: True, 2: False, 3: False, 4: True},
    "MITNETZ":                {1: True, 2: False, 3: False, 4: True},
    "Stadtwerke München":     {1: True, 2: False, 3: False, 4: True},
    "Thüringer Energienetze": {1: True, 2: False, 3: False, 4: True},
    "LEW":                    {1: True, 2: True, 3: True, 4: True},
    "NetzeBW":                {1: True, 2: True, 3: True, 4: True},
    "Bayernwerk":             {1: False, 2: True, 3: True, 4: False},
    "EAM Netz":               {1: True, 2: True, 3: True, 4: True},
}

# Extracted 24-hour tariff pattern from heatmap (same for all DSOs)
# 00–23: NT/ST/HT allocation
MOD3_HOURLY_PATTERN = [
    "NT", "NT", "NT", "NT", "NT",
    "ST", "ST", "ST", "ST", "ST", "ST", "ST", "ST", "ST",
    "HT", "HT", "HT", "HT", "HT", "HT", "HT",
    "ST", "ST", "NT",
]  # length 24

def build_grid_fee_series(dso, num_days):
    """
    Build a full-year 15-min grid fee series (€/kWh) for the selected DSO.
    This just repeats the daily Mod3 pattern; we'll gate by quarter later.
    """
    prices = dso_tariffs[dso]  # ct/kWh
    day_96 = []
    for code in MOD3_HOURLY_PATTERN:
        grid_ct = prices[code]      # ct/kWh
        grid_eur = grid_ct / 100.0  # €/kWh
        day_96.extend([grid_eur] * 4)  # 4×15min per hour
    return np.tile(day_96, num_days)  # length = 96 * num_days

BASE_DATE = datetime.date(2023, 1, 1)

def get_quarter(month: int) -> int:
    if month <= 3:
        return 1
    elif month <= 6:
        return 2
    elif month <= 9:
        return 3
    else:
        return 4

def is_mod3_valid_day(day_idx: int, dso: str) -> bool:
    """Return True if Modul 3 applies on this day for the selected DSO."""
    date = BASE_DATE + datetime.timedelta(days=int(day_idx))
    q = get_quarter(date.month)
    return dso_quarter_valid.get(dso, {}).get(q, False)

def compute_da_indexed_daily_mod3(E, Pmax, quarters, da_day, grid_q_day, taxes, vat):
    """DA-indexed chronological charging with time-varying grid fee."""
    da_q = np.repeat(da_day / 1000.0, 4)  # €/kWh
    remain = E
    cost = 0.0
    emax = Pmax * 0.25  # kWh per quarter

    for q in quarters:
        if remain <= 1e-6:
            break
        e = min(emax, remain)
        price = apply_tariffs(da_q[q], grid_q_day[q], taxes, vat)
        cost += e * price
        remain -= e

    if remain > 1e-6:
        raise ValueError("Charging window too short to deliver requested energy.")
    return cost

def compute_optimised_daily_cost_mod3(E, Pmax, quarters, price_q, grid_q_day, taxes, vat):
    """
    Optimised charging using full cost (wholesale + time-varying grid + taxes).
    """
    pq = price_q / 1000.0  # €/kWh
    emax = Pmax * 0.25
    if len(quarters) * emax < E - 1e-6:
        raise ValueError("Charging window too short for requested energy.")

    full_price = np.array(
        [apply_tariffs(pq[i], grid_q_day[i], taxes, vat) for i in range(len(grid_q_day))]
    )

    sorted_q = sorted(quarters, key=lambda x: full_price[x])

    remain = E
    cost = 0.0
    for q in sorted_q:
        if remain <= 1e-6:
            break
        e = min(emax, remain)
        cost += e * full_price[q]
        remain -= e

    return cost

# =============================================================================
# V2G DAILY OPTIMISATION (SoC-based LP)
# =============================================================================
def compute_v2g_daily_cost(
    battery_kwh,
    soc_start_pct,
    soc_target_pct,
    soc_min_pct,
    p_charge_max,
    p_discharge_max,
    eta_ch,
    eta_dis,
    wholesale_q,
    grid_q,
    taxes,
    vat,
    quarters,
    export_factor=1.0,
    return_profile=False,
):
    """
    Balanced, solver-free V2G optimiser (no pulp / scipy needed).

    - 15-min resolution
    - SoC-based (start %, target %, min %)
    - Bidirectional: charge + discharge
    - 'Balanced' behaviour:
        * charges when prices are cheap or SoC is below target
        * discharges when prices are high enough (spread > threshold)
        * always respects SoC bounds and ability to reach target

    If return_profile=False  -> returns: cost (float, €/day)
    If return_profile=True   -> returns: (cost, e_ch, e_dis, soc_series)
        e_ch, e_dis: kWh per 15-min slot (length Q)
        soc_series: SoC in kWh at each node (length Q+1)
    """

    wholesale_q = np.array(wholesale_q, dtype=float)
    grid_q = np.array(grid_q, dtype=float)
    Q = len(wholesale_q)

    if len(grid_q) != Q:
        raise ValueError("V2G: wholesale and grid arrays must have same length.")

    # Convert SoC to kWh
    soc0 = battery_kwh * soc_start_pct / 100.0
    soc_target = battery_kwh * soc_target_pct / 100.0
    soc_min = battery_kwh * soc_min_pct / 100.0

    if soc_start_pct < soc_min_pct:
        raise ValueError("Arrival SoC (%) must be ≥ minimum SoC (%).")
    if soc_target_pct < soc_min_pct:
        raise ValueError("Target SoC (%) must be ≥ minimum SoC (%).")

    # Prices in €/kWh
    wholesale_kwh = wholesale_q / 1000.0
    import_price = apply_tariffs(wholesale_kwh, grid_q, taxes, vat)
    export_price = export_factor * wholesale_kwh

    # Availability mask
    quarters_set = set(quarters)
    available_mask = np.array([t in quarters_set for t in range(Q)], dtype=bool)
    available_indices = np.where(available_mask)[0]
    n_avail = len(available_indices)

    if n_avail == 0:
        # No time to do anything → just hold SoC, cost 0
        if return_profile:
            soc_series = np.full(Q + 1, soc0)
            e_ch = np.zeros(Q)
            e_dis = np.zeros(Q)
            return 0.0, e_ch, e_dis, soc_series
        return 0.0

    # Power → kWh per quarter
    emax_ch = p_charge_max * 0.25
    emax_dis = p_discharge_max * 0.25

    # Feasibility check: can we reach soc_target at all?
    max_extra_soc = eta_ch * emax_ch * n_avail
    if soc_target - soc0 > max_extra_soc + 1e-6:
        raise ValueError("V2G: impossible to reach target SoC with given window & power.")

    # Precompute: number of future available slots from each t
    future_slots = np.zeros(Q, dtype=int)
    count = 0
    for t in range(Q - 1, -1, -1):
        if available_mask[t]:
            count += 1
        future_slots[t] = count

    # Thresholds for "cheap" vs "expensive"
    import_avail = import_price[available_mask]
    cheap_threshold = np.quantile(import_avail, 0.3)  # lower 30% = cheap

    # Profit spread threshold (balanced mode): ~0.5 ct/kWh
    spread_threshold = 0.005  # €/kWh

    # Decision arrays
    e_ch = np.zeros(Q, dtype=float)
    e_dis = np.zeros(Q, dtype=float)

    soc = soc0

    # ---------- FORWARD PASS: main heuristic ----------
    for t in range(Q):
        if not available_mask[t]:
            continue

        # How many slots left including t?
        n_future = future_slots[t]
        max_future_charge_soc = eta_ch * emax_ch * n_future

        # Small safety factor: assume we'll only manage ~90% of max future charge
        max_future_charge_soc *= 0.9

        # Minimum SoC we can safely go to and still reach target later
        soc_min_allowed = max(soc_min, soc_target - max_future_charge_soc)

        # Price signal
        p_in = import_price[t]
        p_out = export_price[t]
        spread = p_out - p_in

        # 1) Discharge if it's clearly profitable and SoC is above safety floor
        if spread > spread_threshold and soc > soc_min_allowed:
            max_e_dis_by_soc = (soc - soc_min_allowed) * eta_dis
            max_e_dis = min(emax_dis, max_e_dis_by_soc)
            if max_e_dis > 0:
                e_dis[t] = max_e_dis

        # 2) Charge if cheap or we still need SoC to reach target
        soc_after_dis = soc - (1.0 / eta_dis) * e_dis[t]

        need_soc = soc_after_dis < soc_target  # below target
        is_cheap = p_in <= cheap_threshold

        if (need_soc or is_cheap) and soc_after_dis < battery_kwh:
            max_e_ch_by_soc = (battery_kwh - soc_after_dis) / eta_ch
            max_e_ch = min(emax_ch, max_e_ch_by_soc)
            if max_e_ch > 0:
                e_ch[t] = max_e_ch

        # Update SoC for next step
        soc = soc_after_dis + eta_ch * e_ch[t]

    # ---------- SECOND PASS: ensure final SoC ≥ target ----------
    soc = soc0
    for t in range(Q):
        if not available_mask[t]:
            soc_next = soc
        else:
            soc_next = soc - (1.0 / eta_dis) * e_dis[t] + eta_ch * e_ch[t]

            if soc_next < soc_target and e_ch[t] < emax_ch:
                max_extra_power = emax_ch - e_ch[t]
                max_extra_by_cap = 0.0
                if soc_next < battery_kwh:
                    max_extra_by_cap = (battery_kwh - soc_next) / eta_ch

                extra = min(max_extra_power, max_extra_by_cap)
                if extra > 0:
                    e_ch[t] += extra
                    soc_next = soc - (1.0 / eta_dis) * e_dis[t] + eta_ch * e_ch[t]

        soc = soc_next

    if soc < soc_target - 1e-6:
        raise ValueError("V2G: could not reach target SoC given constraints.")

    # ---------- FINAL SOC SERIES ----------
    soc_series = np.zeros(Q + 1, dtype=float)
    soc_series[0] = soc0
    soc = soc0
    for t in range(Q):
        soc = soc - (1.0 / eta_dis) * e_dis[t] + eta_ch * e_ch[t]
        soc_series[t + 1] = soc

    # ---------- COST CALCULATION ----------
    cost = float(np.sum(import_price * e_ch - export_price * e_dis))

    if return_profile:
        return cost, e_ch, e_dis, soc_series
    return cost




# =============================================================================
# FILE LOADER
# =============================================================================
def load_price_series_from_csv(upload, multiple):
    df = pd.read_csv(upload)
    if "price" in df.columns:
        series = df["price"].values
    else:
        series = df.select_dtypes(include="number").iloc[:, 0].values
    if len(series) % multiple != 0:
        st.error(
            f"Uploaded file length {len(series)} is not divisible by {multiple}. "
            "Check that you have full-year data at the correct resolution."
        )
        return None, 0
    return series.astype(float), len(series) // multiple

# =============================================================================
# SIDEBAR SETTINGS
# =============================================================================
st.sidebar.title("Simulation Settings")

st.sidebar.subheader("EV & Charging")
energy = st.sidebar.number_input("Energy per session (kWh)", 1.0, 200.0, 40.0, 1.0)
power = st.sidebar.number_input("Max charging power (kW)", 1.0, 50.0, 11.0, 0.5)

arrival = st.sidebar.slider("Arrival time [h]", 0.0, 24.0, 18.0, 0.25)
departure = st.sidebar.slider("Departure time [h]", 0.0, 24.0, 7.0, 0.25)

st.sidebar.subheader("Charging Frequency")
freq = st.sidebar.selectbox(
    "Pattern",
    [
        "Every day",
        "Every other day",
        "Weekdays only (Mon–Fri)",
        "Weekends only (Sat–Sun)",
        "Custom weekdays",
        "Custom: X sessions per week",
    ],
)
custom_days = None
sessions_week = None
if freq == "Custom weekdays":
    custom_days = st.sidebar.multiselect(
        "Select weekdays",
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        default=["Mon", "Wed", "Fri"],
    )
if freq == "Custom: X sessions per week":
    sessions_week = st.sidebar.number_input("Sessions per week", 1, 14, 3)

st.sidebar.subheader("Tariffs")
flat_price = st.sidebar.number_input("Flat all-in price (€/kWh)", 0.0, 1.0, 0.35, 0.01)
grid = st.sidebar.number_input("Grid network charges (€/kWh)", 0.0, 1.0, 0.11, 0.01)
taxes = st.sidebar.number_input("Taxes & levies (€/kWh)", 0.0, 1.0, 0.05, 0.01)
vat = st.sidebar.number_input("VAT (%)", 0.0, 30.0, 19.0, 1.0)

# =============================================================================
# V2G SETTINGS (SoC-based)
# =============================================================================
st.sidebar.subheader("V2G (Vehicle-to-Grid)")
enable_v2g = st.sidebar.checkbox("Enable V2G (bidirectional charging)", value=False)

if enable_v2g:
    battery_capacity = st.sidebar.number_input(
        "Battery capacity (kWh)",
        min_value=10.0,
        max_value=150.0,
        value=60.0,
    )
    soc_start_pct = st.sidebar.number_input(
        "Arrival SoC (%)",
        min_value=0,
        max_value=100,
        value=40,
    )
    soc_target_pct = st.sidebar.number_input(
        "Required SoC at departure (%)",
        min_value=0,
        max_value=100,
        value=80,
    )
    soc_min_pct = st.sidebar.number_input(
        "Minimum allowed SoC (%)",
        min_value=0,
        max_value=100,
        value=20,
    )

    p_discharge_max = st.sidebar.number_input(
        "Max discharge power (kW)",
        min_value=0.0,
        max_value=100.0,
        value=power,
    )
    eta_ch = st.sidebar.number_input(
        "Charging efficiency",
        min_value=0.5,
        max_value=1.0,
        value=0.95,
    )
    eta_dis = st.sidebar.number_input(
        "Discharging efficiency",
        min_value=0.5,
        max_value=1.0,
        value=0.95,
    )

    export_factor = st.sidebar.number_input(
        "Export price factor (1.0 = wholesale, 0.9 = 90%)",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
    )
else:
    # Safe defaults so variables exist even if V2G is off
    battery_capacity = 60.0
    soc_start_pct = 40
    soc_target_pct = 80
    soc_min_pct = 20
    p_discharge_max = power
    eta_ch = 0.95
    eta_dis = 0.95
    export_factor = 1.0

st.sidebar.subheader("§14a Time-Variable Grid Fee (Modul 3)")
enable_mod3 = st.sidebar.checkbox("Enable time-variable grid fee (§14a EnWG Modul 3)")

selected_dso = None
if enable_mod3:
    selected_dso = st.sidebar.selectbox(
        "Select DSO",
        list(dso_tariffs.keys()),
    )

st.sidebar.subheader("Market Data")
da_file = st.sidebar.file_uploader("DA hourly prices CSV (€/MWh)")
id_file = st.sidebar.file_uploader("ID 15-min prices CSV (€/MWh)")

# =============================================================================
# MARKET DATA LOADING
# =============================================================================
da_day_syn, id_day_syn = get_synthetic_daily_price_profiles()

if da_file is not None:
    da_series, da_days = load_price_series_from_csv(da_file, 24)
else:
    da_series, da_days = np.tile(da_day_syn, 365), 365

if id_file is not None:
    id_series, id_days = load_price_series_from_csv(id_file, 96)
else:
    id_series, id_days = np.tile(id_day_syn, 365), 365

if da_series is None or id_series is None:
    st.stop()

num_days = min(da_days, id_days)
da_year = da_series[: num_days * 24]
id_year = id_series[: num_days * 96]

st.sidebar.info(f"Using **{num_days} days** of DA & ID price data.")

grid_fee_series = None
if enable_mod3 and selected_dso is not None:
    grid_fee_series = build_grid_fee_series(selected_dso, num_days)

# =============================================================================
# OVERVIEW SECTION
# =============================================================================
st.markdown("<h2 id='overview'></h2>", unsafe_allow_html=True)
col_overview, col_snapshot = st.columns([2, 1])

with col_overview:
    st.markdown(
        """
        <div class="eon-card">
            <h3>Overview</h3>
            <p>
            This dashboard compares EV charging cost scenarios over a full year
            using hourly day-ahead and 15-minute intraday prices:
            </p>
            <ul>
                <li><b>Flat retail:</b> constant €/kWh, no market exposure</li>
                <li><b>DA-indexed:</b> hourly wholesale pass-through, no optimisation</li>
                <li><b>DA-optimised:</b> smart charging using DA prices</li>
                <li><b>DA+ID-optimised:</b> smart charging using min(DA, ID)</li>
                <li><b>V2G DA+ID-optimised (optional):</b> SoC-based bi-directional charging & discharging using min(DA, ID)</li>
            </ul>
            <p>
            Optionally, <b>§14a EnWG Modul 3</b> applies a time-variable grid fee
            (HT / ST / NT) by DSO. Modul-3 validity can differ by DSO and quarter
            (Q1–Q4) based on the regulatory design.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_snapshot:
    modul3_state = "ON" if (enable_mod3 and selected_dso) else "OFF"
    v2g_state = "ON" if enable_v2g else "OFF"
    st.markdown(
        f"""
        <div class="eon-card">
            <h3>Scenario Snapshot</h3>
            <p><b>Energy / session:</b> {energy:.1f} kWh</p>
            <p><b>Max power:</b> {power:.1f} kW</p>
            <p><b>Arrival:</b> {arrival:.2f} h</p>
            <p><b>Departure:</b> {departure:.2f} h</p>
            <p><b>Flat price:</b> {flat_price:.2f} €/kWh</p>
            <p><b>§14a Modul 3:</b> {modul3_state}{' – ' + selected_dso if (enable_mod3 and selected_dso) else ''}</p>
            <p><b>V2G:</b> {v2g_state}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# MARKET PRICE CURVES
# =============================================================================
st.markdown("<h2 id='prices'></h2>", unsafe_allow_html=True)
st.markdown('<div class="eon-card">', unsafe_allow_html=True)
st.subheader("Market Price Curves (Full Year)")

timestamps = pd.date_range(start=str(BASE_DATE), periods=num_days * 96, freq="15min")
da_quarter = np.repeat(da_year, 4)[: num_days * 96]
id_quarter = id_year[: num_days * 96]
effective = np.minimum(da_quarter, id_quarter)

df_plot = pd.DataFrame(
    {
        "timestamp": timestamps,
        "DA (€/MWh)": da_quarter,
        "ID (€/MWh)": id_quarter,
        "Effective min(DA, ID) (€/MWh)": effective,
    }
)

fig = px.line(
    df_plot,
    x="timestamp",
    y=["DA (€/MWh)", "ID (€/MWh)", "Effective min(DA, ID) (€/MWh)"],
    title="Day-Ahead & Intraday Prices (zoom & pan)",
)
fig.update_layout(
    xaxis_rangeslider_visible=True,
    height=460,
    legend=dict(orientation="h", y=-0.22),
    plot_bgcolor="#020617",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e5e7eb"),
)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(gridcolor="rgba(148,163,184,0.3)")

st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# COST CALCULATION
# =============================================================================
quarters, _, _ = build_available_quarters(arrival, departure)
charging_days = compute_charging_days(freq, num_days, custom_days, sessions_week)
sessions = len(charging_days)

flat_annual = flat_price * energy * sessions

# Baseline (no Mod3) costs
da_index_annual = 0.0
da_opt_annual = 0.0
da_id_annual = 0.0

# With Mod3 costs
da_index_annual_mod3 = 0.0
da_opt_annual_mod3 = 0.0
da_id_annual_mod3 = 0.0

# V2G annual costs (SoC-based)
v2g_da_id_annual = 0.0
v2g_da_id_mod3_annual = 0.0

try:
    for d in charging_days:
        da_day = da_year[d * 24 : (d + 1) * 24]
        id_day = id_year[d * 96 : (d + 1) * 96]
        da_q_day = np.repeat(da_day, 4)
        eff_day = np.minimum(da_q_day, id_day)

        # --- Baseline constant grid fee ---
        c_da_index = compute_da_indexed_baseline_daily(
            energy, power, quarters, da_day, grid, taxes, vat
        )
        c_da_opt = compute_optimised_daily_cost(
            energy, power, quarters, da_q_day, grid, taxes, vat
        )
        c_da_id = compute_optimised_daily_cost(
            energy, power, quarters, eff_day, grid, taxes, vat
        )

        da_index_annual += c_da_index
        da_opt_annual += c_da_opt
        da_id_annual += c_da_id

        # --- V2G, SoC-based, without Modul 3 (constant grid) ---
        if enable_v2g:
            grid_q_const = np.full(96, grid)
            c_v2g_const = compute_v2g_daily_cost(
                battery_kwh=battery_capacity,
                soc_start_pct=soc_start_pct,
                soc_target_pct=soc_target_pct,
                soc_min_pct=soc_min_pct,
                p_charge_max=power,
                p_discharge_max=p_discharge_max,
                eta_ch=eta_ch,
                eta_dis=eta_dis,
                wholesale_q=eff_day,
                grid_q=grid_q_const,
                taxes=taxes,
                vat=vat,
                quarters=quarters,
                export_factor=export_factor,
            )
            v2g_da_id_annual += c_v2g_const
        else:
            c_v2g_const = 0.0

        # --- With Modul 3 (if enabled and valid in this quarter for this DSO) ---
        if enable_mod3 and grid_fee_series is not None and selected_dso is not None and is_mod3_valid_day(d, selected_dso):
            grid_q_day = grid_fee_series[d * 96 : (d + 1) * 96]

            c_da_index_mod3 = compute_da_indexed_daily_mod3(
                energy, power, quarters, da_day, grid_q_day, taxes, vat
            )
            c_da_opt_mod3 = compute_optimised_daily_cost_mod3(
                energy, power, quarters, da_q_day, grid_q_day, taxes, vat
            )
            c_da_id_mod3 = compute_optimised_daily_cost_mod3(
                energy, power, quarters, eff_day, grid_q_day, taxes, vat
            )

            if enable_v2g:
                c_v2g_mod3 = compute_v2g_daily_cost(
                    battery_kwh=battery_capacity,
                    soc_start_pct=soc_start_pct,
                    soc_target_pct=soc_target_pct,
                    soc_min_pct=soc_min_pct,
                    p_charge_max=power,
                    p_discharge_max=p_discharge_max,
                    eta_ch=eta_ch,
                    eta_dis=eta_dis,
                    wholesale_q=eff_day,
                    grid_q=grid_q_day,
                    taxes=taxes,
                    vat=vat,
                    quarters=quarters,
                    export_factor=export_factor,
                )
            else:
                c_v2g_mod3 = c_v2g_const
        else:
            # Not in a Mod3-valid quarter or feature off:
            c_da_index_mod3 = c_da_index
            c_da_opt_mod3 = c_da_opt
            c_da_id_mod3 = c_da_id
            c_v2g_mod3 = c_v2g_const

        da_index_annual_mod3 += c_da_index_mod3
        da_opt_annual_mod3 += c_da_opt_mod3
        da_id_annual_mod3 += c_da_id_mod3
        v2g_da_id_mod3_annual += c_v2g_mod3

except ValueError as e:
    st.error(str(e))
# =============================================================================
# V2G DAILY PROFILE (for visual inspection)
# =============================================================================
v2g_profile_data = None  # (day_idx, time_hours, e_ch_kW, e_dis_kW, soc_series)

if enable_v2g and charging_days:
    try:
        # Use the FIRST charging day as example
        example_day = int(charging_days[0])

        da_day_ex = da_year[example_day * 24 : (example_day + 1) * 24]
        id_day_ex = id_year[example_day * 96 : (example_day + 1) * 96]
        da_q_ex = np.repeat(da_day_ex, 4)
        eff_ex = np.minimum(da_q_ex, id_day_ex)

        # Choose grid fee for that day (Modul 3 if active & valid, else flat grid)
        if enable_mod3 and grid_fee_series is not None and selected_dso is not None and is_mod3_valid_day(example_day, selected_dso):
            grid_q_ex = grid_fee_series[example_day * 96 : (example_day + 1) * 96]
        else:
            grid_q_ex = np.full(96, grid)

        # Compute detailed profile
        cost_ex, e_ch_ex, e_dis_ex, soc_ex = compute_v2g_daily_cost(
            battery_kwh=battery_capacity,
            soc_start_pct=soc_start_pct,
            soc_target_pct=soc_target_pct,
            soc_min_pct=soc_min_pct,
            p_charge_max=power,
            p_discharge_max=p_discharge_max,
            eta_ch=eta_ch,
            eta_dis=eta_dis,
            wholesale_q=eff_ex,
            grid_q=grid_q_ex,
            taxes=taxes,
            vat=vat,
            quarters=quarters,
            export_factor=export_factor,
            return_profile=True,
        )

        # Convert energy (kWh per 15-min) to power (kW) for plotting
        e_ch_kW = e_ch_ex * 4.0
        e_dis_kW = e_dis_ex * 4.0

        time_hours = np.arange(96) / 4.0  # 0..24h in 0.25h steps

        v2g_profile_data = (example_day, time_hours, e_ch_kW, e_dis_kW, soc_ex)

    except Exception as e:
        st.warning(f'Could not generate V2G daily profile: {e}')

# =============================================================================
# RESULTS – 3 BLOCKS + BAR CHARTS
# =============================================================================
st.markdown("<h2 id='results'></h2>", unsafe_allow_html=True)
st.markdown('<div class="eon-card">', unsafe_allow_html=True)

# 1) Annual costs WITHOUT Modul 3
st.subheader("1️⃣ Annual Cost (WITHOUT §14a Modul 3)")

scenarios_before = [
    "Flat retail (fixed price)",
    "DA-indexed",
    "DA-optimised",
    "DA+ID-optimised",
]
costs_before = [
    round(flat_annual),
    round(da_index_annual),
    round(da_opt_annual),
    round(da_id_annual),
]

if enable_v2g:
    scenarios_before.append("V2G DA+ID-optimised (SoC-based)")
    costs_before.append(round(v2g_da_id_annual))

df_before = pd.DataFrame(
    {
        "Scenario": scenarios_before,
        "Annual cost (€)": costs_before,
    }
)
st.table(df_before)

# 2) Annual costs WITH Modul 3
if enable_mod3 and grid_fee_series is not None and selected_dso is not None:
    st.subheader("2️⃣ Annual Cost (WITH §14a Modul 3 – by DSO & quarter)")

    scenarios_after = [
        "Flat retail (unchanged)",
        f"DA-indexed + Modul 3 ({selected_dso})",
        f"DA-optimised + Modul 3 ({selected_dso})",
        f"DA+ID-optimised + Modul 3 ({selected_dso})",
    ]
    costs_after = [
        round(flat_annual),
        round(da_index_annual_mod3),
        round(da_opt_annual_mod3),
        round(da_id_annual_mod3),
    ]

    if enable_v2g:
        scenarios_after.append(f"V2G DA+ID + Modul 3 ({selected_dso})")
        costs_after.append(round(v2g_da_id_mod3_annual))

    df_after = pd.DataFrame(
        {
            "Scenario": scenarios_after,
            "Annual cost (€)": costs_after,
        }
    )
    st.table(df_after)

    # 3) Modul 3 incremental savings
    st.subheader("3️⃣ Additional Savings FROM §14a Modul 3 (vs constant grid fee)")

    scenarios_mod3 = [
        "DA-indexed → Modul 3",
        "DA-optimised → Modul 3",
        "DA+ID-optimised → Modul 3",
    ]
    savings_mod3 = [
        round(da_index_annual - da_index_annual_mod3),
        round(da_opt_annual - da_opt_annual_mod3),
        round(da_id_annual - da_id_annual_mod3),
    ]

    if enable_v2g:
        scenarios_mod3.append("V2G DA+ID → V2G DA+ID + Modul 3")
        savings_mod3.append(round(v2g_da_id_annual - v2g_da_id_mod3_annual))

    df_mod3 = pd.DataFrame(
        {
            "Scenario": scenarios_mod3,
            "Extra savings (€ / year)": savings_mod3,
        }
    )
    st.table(df_mod3)
# -----------------------------
# ENHANCED V2G DAILY PROFILE PLOT (3-panel diagnostic)
# -----------------------------
if enable_v2g and v2g_profile_data is not None:

    example_day_idx, time_hours, e_ch_kW, e_dis_kW, soc_ex = v2g_profile_data

    st.markdown("### 🔍 Enhanced V2G Daily Profile — Charge/Discharge, SoC, and Price")
    st.markdown(f"*Example day: **{example_day_idx + 1}** of the simulation year*")

    # Create 3-row subplot
    fig_v2g = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Charging & Discharging Power (kW)",
            "State of Charge (kWh)",
            "Import Price (€/kWh)"
        )
    )

    # ---------------------------------------------------------
    # 1. SHADED REGIONS FOR EV CONNECTED TIME (across all panels)
    # ---------------------------------------------------------
    for t in range(96):
        # Only shade where EV is connected (quarters comes from main model)
        if t in quarters:
            start = t / 4
            end = (t + 1) / 4
            fig_v2g.add_vrect(
                x0=start, x1=end,
                fillcolor="rgba(0,150,255,0.10)",
                line_width=0,
                row="all", col=1
            )

    # ---------------------------------------------------------
    # 2. PANEL 1 – Charge & Discharge Power
    # ---------------------------------------------------------
    fig_v2g.add_trace(
        go.Bar(
            x=time_hours,
            y=e_ch_kW,
            name="Charge (kW)",
            marker_color="#4DA3FF",
            opacity=0.9
        ),
        row=1, col=1
    )

    fig_v2g.add_trace(
        go.Bar(
            x=time_hours,
            y=-e_dis_kW,
            name="Discharge (kW)",
            marker_color="#FF6A6A",
            opacity=0.9
        ),
        row=1, col=1
    )

    fig_v2g.update_yaxes(title_text="Power (kW)", row=1, col=1)

    # ---------------------------------------------------------
    # 3. PANEL 2 – SoC (kWh)
    # ---------------------------------------------------------
    fig_v2g.add_trace(
        go.Scatter(
            x=time_hours,
            y=soc_ex[:-1],
            name="SoC (kWh)",
            line=dict(color="#FFB070", width=3),
            mode="lines+markers"
        ),
        row=2, col=1
    )
    fig_v2g.update_yaxes(title_text="SoC (kWh)", row=2, col=1)

    # ---------------------------------------------------------
    # 4. PANEL 3 – Import Price (€/kWh)
    # ---------------------------------------------------------

    # Compute €/kWh import price curve for the example day
    import_price_ex = apply_tariffs(
        (eff_ex / 1000.0),
        grid_q_ex,
        taxes,
        vat
    )

    fig_v2g.add_trace(
        go.Scatter(
            x=time_hours,
            y=import_price_ex,
            name="Import Price (€/kWh)",
            line=dict(color="#00FFAA", width=2),
        ),
        row=3, col=1
    )
    fig_v2g.update_yaxes(title_text="€/kWh", row=3, col=1)

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------
    fig_v2g.update_layout(
        barmode="relative",
        height=900,
        showlegend=True,
        plot_bgcolor="#020617",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        legend=dict(orientation="h", y=-0.2),
        xaxis3_title="Time of Day (hours)",
    )

    st.plotly_chart(fig_v2g, use_container_width=True)

elif enable_v2g:
    st.warning("⚠️ V2G enabled, but profile data could not be generated.")


# =============================================================================
# AIX ASSISTANT — MODEL-AWARE ENGINE (FIXED & IMPROVED FOR GROQ)
# =============================================================================
import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def aix_answer(user_message):
    """AIX assistant that ALWAYS uses your model outputs when answering."""

    # ---- Build explicit, machine-readable context ----
    try:
        model_context = f"""
        === EV MODEL RESULTS ===

        DA_INDEXED:
            WITHOUT_M3: {da_index_annual:.2f}
            WITH_M3:    {da_index_annual_mod3:.2f}
            SAVINGS:    {da_index_annual - da_index_annual_mod3:.2f}

        DA_OPTIMISED:
            WITHOUT_M3: {da_opt_annual:.2f}
            WITH_M3:    {da_opt_annual_mod3:.2f}
            SAVINGS:    {da_opt_annual - da_opt_annual_mod3:.2f}

        DA_ID_OPTIMISED:
            WITHOUT_M3: {da_id_annual:.2f}
            WITH_M3:    {da_id_annual_mod3:.2f}
            SAVINGS:    {da_id_annual - da_id_annual_mod3:.2f}

        V2G_DA_ID_OPTIMISED:
            ENABLED:    {enable_v2g}
            WITHOUT_M3: {v2g_da_id_annual:.2f}
            WITH_M3:    {v2g_da_id_mod3_annual:.2f}
            SAVINGS:    {v2g_da_id_annual - v2g_da_id_mod3_annual:.2f}

        === RULES ===
        - Modul 3 only gives reduced grid fees during DSO low-load windows.
        - DA-indexed does NOT shift load → often misses those hours → can have negative savings.
        - DA-optimised & DA+ID-optimised shift charging → benefit strongly from Modul 3.
        - V2G DA+ID uses SoC-based optimisation, not fixed kWh/session.
        """

    except Exception:
        model_context = "MODEL_RESULTS_NOT_AVAILABLE"

    # ---- Strong system instructions ----
    system_prompt = f"""
    You are AIX, the expert assistant for EV smart charging, V2G & §14a Modul 3.

    You ALWAYS use the model results given in the section '=== EV MODEL RESULTS ==='.  
    You NEVER say “no results provided” or “I don't know the results”.  
    If the user asks vague things like “Explain results”, interpret it as:
        → “Explain the model results in the provided context”.

    Your job:
      - Explain the scenarios (flat, DA-indexed, DA-optimised, DA+ID-optimised, V2G)
      - Interpret costs, savings, differences
      - Explain why savings are positive or negative
      - Be specific, numeric, and concise
      - ALWAYS reference real values provided in the context

    Here is the model context you must use:

    {model_context}
    """

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }

    payload = {
        "model": "llama-3.3-70b-versatile",    # Latest supported
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.2
    }

    try:
        r = requests.post(url, headers=headers, json=payload)
        data = r.json()

        if "error" in data:
            return "⚠️ API Error: " + data["error"].get("message", "")

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"⚠️ Request failed: {str(e)}"



    # -----------------------------
    # BAR CHARTS FOR RESULTS
    # -----------------------------
    st.markdown("### 📊 Annual Cost Comparison (Before vs After Modul 3)")

    scenarios = ["DA-indexed", "DA-optimised", "DA+ID-optimised"]
    before_vals = [da_index_annual, da_opt_annual, da_id_annual]
    after_vals = [da_index_annual_mod3, da_opt_annual_mod3, da_id_annual_mod3]

    fig_cost = go.Figure(
        data=[
            go.Bar(name="Before Modul 3", x=scenarios, y=before_vals),
            go.Bar(name="With Modul 3", x=scenarios, y=after_vals),
        ]
    )
    fig_cost.update_layout(
        barmode="group",
        height=450,
        plot_bgcolor="#020617",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        yaxis_title="Annual Cost (€)",
        xaxis_title="Customer Type",
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    st.markdown("### 📉 Additional Savings from Modul 3")

    savings_mod3 = [
        da_index_annual - da_index_annual_mod3,
        da_opt_annual - da_opt_annual_mod3,
        da_id_annual - da_id_annual_mod3,
    ]

    fig_savings = go.Figure(
        data=[go.Bar(x=scenarios, y=savings_mod3)]
    )
    fig_savings.update_layout(
        height=400,
        plot_bgcolor="#020617",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        yaxis_title="Modul 3 Extra Savings (€ / year)",
        xaxis_title="Customer Type",
    )
    st.plotly_chart(fig_savings, use_container_width=True)


st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# DETAILS
# =============================================================================
st.markdown("<h2 id='details'></h2>", unsafe_allow_html=True)
st.markdown('<div class="eon-card">', unsafe_allow_html=True)
st.subheader("Model Assumptions & Notes")
st.markdown(
    f"""
- Resolution: **15 minutes (96 slots per day)**  
- DA prices: **hourly** in €/MWh; ID prices: **15-min** in €/MWh  
- If no CSV files are uploaded, synthetic DA/ID profiles are repeated over {num_days} days.  
- Effective wholesale price for DA+ID optimisation: **min(DA, ID)** per 15-min slot.  
- Wholesale-based customers pay:  
  **(wholesale energy price + grid network charges + taxes & levies) × (1 + VAT)**  
- Flat retail case: all-in constant €/kWh (no extra VAT applied in the model).  
- Charging pattern: **{freq}**, giving **{sessions} sessions** across {num_days} days.  
- §14a EnWG Modul 3:  
  - Uses DSO-specific **HT/ST/NT grid prices (ct/kWh)**.  
  - Hourly pattern from the graphic is applied with 15-min resolution.  
  - Modul-3 validity by DSO & quarter is configurable and currently set as you provided.  
- V2G mode (when enabled):  
  - Optimisation is **SoC-based**, not fixed energy/session.  
  - Battery capacity: **{battery_capacity:.1f} kWh**, SoC arrival: **{soc_start_pct}%**, target: **{soc_target_pct}%**, minimum: **{soc_min_pct}%**.  
  - Charge/discharge limited by max power and round-trip efficiencies you configured.  
"""
)
st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# AIX ASSISTANT — CHAT SESSION LOGIC (STEP 2)
# =============================================================================

if "aix_history" not in st.session_state:
    st.session_state.aix_history = []

# Display chat messages
for role, msg in st.session_state.aix_history:
    if role == "user":
        st.markdown(
            f"<p style='color:#FF4B4B; font-weight:600;'>🧑‍💬 You: {msg}</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<p style='color:white;'><span style='color:#4DA6FF; font-weight:700;'>🤖 AIX:</span> {msg}</p>",
            unsafe_allow_html=True
        )

# Input bar
user_msg = st.chat_input("Ask AIX about results, savings, tariffs, optimisations…")

if user_msg:
    st.session_state.aix_history.append(("user", user_msg))
    reply = aix_answer(user_msg)  # <— uses Groq now
    st.session_state.aix_history.append(("assistant", reply))
    st.rerun()

# =============================================================================
# AIX ASSISTANT — FLOATING CHAT BUBBLE UI (STEP 3)
# =============================================================================
st.markdown("""
<style>
#aix_chat_button {
    position: fixed;
    bottom: 24px;
    right: 24px;
    padding: 12px 22px;
    background-color: #E2000F;
    color: white;
    border-radius: 40px;
    font-weight: 600;
    cursor: pointer;
    z-index: 9999;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.45);
}
#aix_chat_window {
    position: fixed;
    bottom: 90px;
    right: 24px;
    width: 350px;
    height: 480px;
    background-color: #0d1117;
    border-radius: 12px;
    padding: 18px;
    border: 2px solid #E2000F;
    overflow-y: auto;
    display: none;
    z-index: 10000;
}
</style>

<div id="aix_chat_button"
     onclick="document.getElementById('aix_chat_window').style.display='block';">
🤖 AIX Assistant
</div>

<div id="aix_chat_window">
    <h3 style="color:#E2000F;margin-top:0;">AIX Assistant</h3>
</div>
""", unsafe_allow_html=True)
