# ============================================================
#  BOT TRIAL APP v6
#  Strict polarity (L vs H must disagree by 2) + no fade after
#  CHoCH/BOS through the zone. Same rules every session.
# ============================================================
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="🤖 S&R Auto-Trader Bot v6")

TRADE_AMT = 200.0
SWING_K = 5
ZONE_TOL = 0.0012
MIN_DRAW_TOUCHES = 2
MIN_TRADE_TOUCHES = 3
MIN_RR = 1.2
COOLDOWN_BARS = 15
BOS_FLAT_COOLDOWN = 5
MIN_SWING_PCT = 0.0008
WARMUP_BARS = 30
STRONG_ZONE = 3
STOP_NOISE_MULT = 1.0
NEAR_ZONE_MULT = 1.2
CHOCH_ENABLE = True
STRUCT_TOUCHES = 3
POLARITY_EDGE = 2          # H must beat L by this to be a shortable ceiling

C_HH = dict(fg="#ffffff", bg="#166534")
C_LH = dict(fg="#ffffff", bg="#991b1b")
C_H  = dict(fg="#ffffff", bg="#334155")
C_BOS_UP = dict(fg="#111111", bg="#f5c518")
C_BOS_DN = dict(fg="#ffffff", bg="#b91c1c")
C_CH_UP  = dict(fg="#111111", bg="#7dd3fc")
C_CH_DN  = dict(fg="#111111", bg="#fb923c")
C_FLOOR  = "#166534"
C_CEIL   = "#991b1b"
C_BROKEN = "#78716c"

defaults = {
    "active": False, "balance": 1000.0, "shares": 0.0,
    "entry": None, "sl": None, "tp": None, "side": None,
    "step": 0, "df": pd.DataFrame(), "log": [], "start_idx": 0,
    "cooldown": 0, "markers": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_data(ttl=3600)
def fetch_session(ticker, day):
    start = pd.Timestamp(day)
    end = start + timedelta(days=1)
    d = yf.download(
        ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
        interval="1m", auto_adjust=True, progress=False, prepost=False,
    )
    if d is None or d.empty:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    df = d.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    tod = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df = df[(tod >= 9 * 60 + 30) & (tod < 16 * 60)].copy()
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.dropna(subset=["close"]).drop_duplicates(subset="timestamp")
    df = df[df["timestamp"].dt.date == day].copy()
    df["label"] = df["timestamp"].dt.strftime("%H:%M")
    return df.reset_index(drop=True)

def _raw_swings(df):
    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    ref = float(df["close"].iloc[-1])
    min_size = max(ref * MIN_SWING_PCT, 1e-6)
    raw = []
    for i in range(SWING_K, n - SWING_K):
        wh = highs[i - SWING_K: i + SWING_K + 1]
        wl = lows[i - SWING_K: i + SWING_K + 1]
        if highs[i] == wh.max() and (highs[i] - wl.min()) >= min_size:
            raw.append((i, float(highs[i]), "H"))
        if lows[i] == wl.min() and (wh.max() - lows[i]) >= min_size:
            raw.append((i, float(lows[i]), "L"))
    raw.sort(key=lambda x: x[0])
    cleaned = []
    for s in raw:
        if cleaned and cleaned[-1][2] == s[2]:
            if (s[2] == "H" and s[1] >= cleaned[-1][1]) or \
               (s[2] == "L" and s[1] <= cleaned[-1][1]):
                cleaned[-1] = s
        else:
            cleaned.append(s)
    labeled, last_h, last_l = [], None, None
    for idx, price, typ in cleaned:
        if typ == "H":
            lab = "H" if last_h is None else ("HH" if price > last_h else "LH")
            last_h = price
        else:
            lab = "L" if last_l is None else ("HL" if price > last_l else "LL")
            last_l = price
        labeled.append({"i": idx, "price": price, "type": typ, "label": lab})
    return labeled

def build_zones(labeled, ref_price):
    zones = []
    for s in labeled:
        placed = False
        for z in zones:
            if abs(s["price"] - z["mid"]) / max(ref_price, 1e-9) < ZONE_TOL:
                z["prices"].append(s["price"])
                z["mid"] = float(np.mean(z["prices"]))
                z["touches"] += 1
                if s["type"] == "L":
                    z["low_touches"] += 1
                else:
                    z["high_touches"] += 1
                placed = True
                break
        if not placed:
            zones.append({
                "mid": s["price"], "prices": [s["price"]], "touches": 1,
                "low_touches": 1 if s["type"] == "L" else 0,
                "high_touches": 1 if s["type"] == "H" else 0,
            })
    out = [z for z in zones if z["touches"] >= MIN_DRAW_TOUCHES]
    for z in out:
        z["min_px"] = min(z["prices"])
        z["max_px"] = max(z["prices"])
    return out

def _inside_strong_floor(price, zones, noise):
    buf = max(noise, (abs(price) * 0.0004) if price else 0.01)
    for z in zones:
        if z["low_touches"] >= STRUCT_TOUCHES and z["low_touches"] >= z["high_touches"] + POLARITY_EDGE:
            if price >= z["min_px"] - buf:
                if abs(price - z["mid"]) <= max(4 * buf, z["max_px"] - z["min_px"] + buf):
                    return True
    return False

def _inside_strong_ceiling(price, zones, noise):
    buf = max(noise, (abs(price) * 0.0004) if price else 0.01)
    for z in zones:
        if z["high_touches"] >= STRUCT_TOUCHES and z["high_touches"] >= z["low_touches"] + POLARITY_EDGE:
            if price <= z["max_px"] + buf:
                if abs(price - z["mid"]) <= max(4 * buf, z["max_px"] - z["min_px"] + buf):
                    return True
    return False

def detect_structure(df):
    if df is None or len(df) < SWING_K * 2 + 1:
        return [], [], []
    labeled = _raw_swings(df)
    n = len(df)
    bos, choch = [], []
    last_hh = last_hl = last_lh = last_ll = None
    bias = "NEUTRAL"
    ptr = 0
    noise_series = (df["high"] - df["low"])
    for i in range(n):
        noise = float(noise_series.iloc[max(0, i - 20): i + 1].mean() or 0.3)
        conf_now = [s for s in labeled if s["i"] + SWING_K <= i]
        zones_now = build_zones(conf_now, float(df["close"].iloc[i]))
        while ptr < len(labeled) and labeled[ptr]["i"] + SWING_K <= i:
            s = labeled[ptr]
            if s["label"] == "HH":
                if not _inside_strong_ceiling(s["price"], zones_now, noise):
                    last_hh = s["price"]
            elif s["label"] == "HL":
                last_hl = s["price"]
            elif s["label"] == "LH":
                last_lh = s["price"]
            elif s["label"] == "LL":
                if not _inside_strong_floor(s["price"], zones_now, noise):
                    last_ll = s["price"]
            ptr += 1
        c = float(df["close"].iloc[i])
        if last_hh is not None and c > last_hh:
            if bias != "BULL":
                bos.append({"i": i, "price": last_hh, "dir": "up"})
                bias = "BULL"
            last_hh = None
            continue
        if last_ll is not None and c < last_ll:
            if bias != "BEAR":
                bos.append({"i": i, "price": last_ll, "dir": "down"})
                bias = "BEAR"
            last_ll = None
            continue
        if CHOCH_ENABLE:
            if bias == "BULL" and last_hl is not None and c < last_hl:
                choch.append({"i": i, "price": last_hl, "dir": "down"})
                last_hl = None
                bias = "NEUTRAL"
            elif bias == "BEAR" and last_lh is not None and c > last_lh:
                choch.append({"i": i, "price": last_lh, "dir": "up"})
                last_lh = None
                bias = "NEUTRAL"
    return labeled, bos, choch

def trend_state(bos, choch, i, labeled=None):
    events = []
    for b in bos:
        if b["i"] <= i:
            events.append((b["i"], "BOS", b["dir"]))
    for c in choch:
        if c["i"] <= i:
            events.append((c["i"], "CHOCH", c["dir"]))
    if events:
        events.sort(key=lambda x: x[0])
        kind, direction = events[-1][1], events[-1][2]
        if kind == "BOS":
            return "BULL" if direction == "up" else "BEAR"
        return "NEUTRAL"
    if not labeled:
        return "NEUTRAL"
    confirmed = [s for s in labeled if s["i"] + SWING_K <= i]
    labs = [s["label"] for s in confirmed[-6:] if s["label"] in ("HH", "HL", "LH", "LL")]
    bull = labs.count("HH") + labs.count("HL")
    bear = labs.count("LH") + labs.count("LL")
    if bull >= 3 and bull > bear:
        return "BULL"
    if bear >= 3 and bear > bull:
        return "BEAR"
    return "NEUTRAL"

def zone_role(z, close, noise):
    """Display role. Trade role is stricter (see tradeable_*)."""
    if z["low_touches"] >= z["high_touches"] + POLARITY_EDGE:
        role = "floor"
    elif z["high_touches"] >= z["low_touches"] + POLARITY_EDGE:
        role = "ceiling"
    else:
        role = "both"
    buffer = max(noise, z["mid"] * 0.0004)
    if role == "floor" and close < z["mid"] - buffer:
        return "broken_floor"
    if role == "ceiling" and close > z["mid"] + buffer:
        return "broken_ceiling"
    return role

def zone_broken_by_close(z, close, noise):
    buffer = max(noise, z["mid"] * 0.0004)
    if close > z["max_px"] + buffer:
        return "up"
    if close < z["min_px"] - buffer:
        return "down"
    return None

def event_through_zone(z, bos, choch, i, noise):
    """True if a CHoCH or BOS already printed through this zone (accepted break)."""
    buf = max(noise, z["mid"] * 0.0004)
    lo, hi = z["min_px"] - buf, z["max_px"] + buf
    for b in bos:
        if b["i"] <= i and lo <= b["price"] <= hi:
            return b["dir"]
    for h in choch:
        if h["i"] <= i and lo <= h["price"] <= hi:
            return h["dir"]
    return None

def classify_zones(zones, price, noise):
    floors, ceilings, boths = [], [], []
    for z in zones:
        role = zone_role(z, price, noise)
        z = dict(z)
        z["role"] = role
        if role == "floor":
            floors.append(z)
        elif role == "ceiling":
            ceilings.append(z)
        elif role == "both":
            boths.append(z)
        elif role == "broken_floor":
            ceilings.append(z)
        elif role == "broken_ceiling":
            floors.append(z)
    return floors, ceilings, boths

def pick_near(zones, price, near, min_touches):
    cand = [z for z in zones
            if z["touches"] >= min_touches and abs(z["mid"] - price) <= near]
    if not cand:
        return None
    cand.sort(key=lambda z: (-z["touches"], abs(z["mid"] - price)))
    return cand[0]

def nearest_below(zs, price, min_touches=MIN_DRAW_TOUCHES):
    below = [z for z in zs if z["mid"] < price and z["touches"] >= min_touches]
    return max(below, key=lambda z: z["mid"]) if below else None

def nearest_above(zs, price, min_touches=MIN_DRAW_TOUCHES):
    above = [z for z in zs if z["mid"] > price and z["touches"] >= min_touches]
    return min(above, key=lambda z: z["mid"]) if above else None

def candle_signal(df, i):
    if i < 1:
        return None
    o, h, l, c = (df["open"].iloc[i], df["high"].iloc[i],
                   df["low"].iloc[i], df["close"].iloc[i])
    po, pc = df["open"].iloc[i - 1], df["close"].iloc[i - 1]
    body = abs(c - o)
    up_wick = h - max(o, c)
    dn_wick = min(o, c) - l
    if pc < po and c > o and c >= po and o <= pc:
        return "bull"
    if dn_wick > 2 * body and up_wick < body and c >= o:
        return "bull"
    if pc > po and c < o and c <= po and o >= pc:
        return "bear"
    if up_wick > 2 * body and dn_wick < body and c <= o:
        return "bear"
    if i >= 2:
        b1 = df["close"].iloc[i - 1] - df["open"].iloc[i - 1]
        b2 = c - o
        avg = (df["high"] - df["low"]).iloc[max(0, i - 20):i].mean()
        if avg and b1 > 0.3 * avg and b2 > 0.3 * avg:
            return "bull"
        if avg and b1 < -0.3 * avg and b2 < -0.3 * avg:
            return "bear"
    return None

def bot_decide(df, i, labeled, bos, choch, zones):
    price = float(df["close"].iloc[i])
    noise = float((df["high"] - df["low"]).iloc[max(0, i - 20): i + 1].mean())
    if not noise or noise <= 0:
        return None
    trend = trend_state(bos, choch, i, labeled)
    sig = candle_signal(df, i)
    if sig is None:
        return None

    floors, ceilings, boths = classify_zones(zones, price, noise)
    near = NEAR_ZONE_MULT * noise

    # Mixed BOTH nearby → skip (mid-range / contested)
    both_here = pick_near(boths, price, near, MIN_TRADE_TOUCHES)
    if both_here is not None:
        return None

    floor_here = pick_near(floors, price, near, MIN_TRADE_TOUCHES)
    ceil_here = pick_near(ceilings, price, near, MIN_TRADE_TOUCHES)

    if floor_here is not None and ceil_here is not None:
        d_f = abs(price - floor_here["mid"])
        d_c = abs(price - ceil_here["mid"])
        if d_f < d_c * 0.6:
            ceil_here = None
        elif d_c < d_f * 0.6:
            floor_here = None
        else:
            return None

    # ----- LONG -----
    if sig == "bull" and floor_here is not None and ceil_here is None:
        # already broken down through this floor → do not long it
        thru = event_through_zone(floor_here, bos, choch, i, noise)
        brk = zone_broken_by_close(floor_here, price, noise)
        if thru == "down" or brk == "down":
            return None
        if floor_here["low_touches"] < floor_here["high_touches"] + POLARITY_EDGE:
            return None
        strong_bottom = floor_here["low_touches"] >= STRONG_ZONE
        allow = (trend != "BEAR") or strong_bottom
        if allow:
            target_pool = ceilings + boths
            target = nearest_above(target_pool, price, MIN_DRAW_TOUCHES)
            if target is not None:
                slp = floor_here["min_px"] - max(STOP_NOISE_MULT * noise, price * 0.0004)
                tpp = target["mid"] - 0.4 * noise
                risk, reward = price - slp, tpp - price
                if risk > 0 and reward / risk >= MIN_RR:
                    tag = "double-bottom override" if trend == "BEAR" else trend
                    return {
                        "side": "LONG", "sl": round(slp, 2), "tp": round(tpp, 2),
                        "why": (f"{tag} | FLOOR {floor_here['mid']:.2f} "
                                f"({floor_here['low_touches']}L/{floor_here['touches']}x) "
                                f"| RR {reward/risk:.1f}"),
                    }

    # ----- SHORT -----
    if sig == "bear" and ceil_here is not None and floor_here is None:
        thru = event_through_zone(ceil_here, bos, choch, i, noise)
        brk = zone_broken_by_close(ceil_here, price, noise)
        # CHoCH↑ / BOS↑ through this ceiling → do not fade it
        if thru == "up" or brk == "up":
            return None
        if ceil_here["high_touches"] < ceil_here["low_touches"] + POLARITY_EDGE:
            return None
        strong_top = ceil_here["high_touches"] >= STRONG_ZONE
        allow = (trend != "BULL") or strong_top
        if allow:
            target_pool = floors + boths
            target = nearest_below(target_pool, price, MIN_DRAW_TOUCHES)
            if target is not None:
                slp = ceil_here["max_px"] + max(STOP_NOISE_MULT * noise, price * 0.0004)
                tpp = target["mid"] + 0.4 * noise
                risk, reward = slp - price, price - tpp
                if risk > 0 and reward / risk >= MIN_RR:
                    tag = "double-top override" if trend == "BULL" else trend
                    return {
                        "side": "SHORT", "sl": round(slp, 2), "tp": round(tpp, 2),
                        "why": (f"{tag} | CEILING {ceil_here['mid']:.2f} "
                                f"({ceil_here['high_touches']}H/{ceil_here['touches']}x) "
                                f"| RR {reward/risk:.1f}"),
                    }
    return None

def _close_long(px, t, kind, reason=""):
    sh = st.session_state.shares
    st.session_state.balance += sh * px
    if kind == "SL":
        st.session_state.log.append(f"{t}: 🛑 LONG stopped ${px:.2f}")
        st.session_state.cooldown = COOLDOWN_BARS
    elif kind == "TP":
        st.session_state.log.append(f"{t}: 🎯 LONG target ${px:.2f}")
    else:
        st.session_state.log.append(f"{t}: ⚡ LONG flattened ${px:.2f} — {reason}")
        st.session_state.cooldown = BOS_FLAT_COOLDOWN
    st.session_state.markers.append((t, px, kind))
    st.session_state.shares = 0
    st.session_state.sl = st.session_state.tp = st.session_state.entry = None
    st.session_state.side = None

def _close_short(px, t, kind, reason=""):
    sh = abs(st.session_state.shares)
    st.session_state.balance -= sh * px
    if kind == "SL":
        st.session_state.log.append(f"{t}: 🛑 SHORT stopped ${px:.2f}")
        st.session_state.cooldown = COOLDOWN_BARS
    elif kind == "TP":
        st.session_state.log.append(f"{t}: 🎯 SHORT covered ${px:.2f}")
    else:
        st.session_state.log.append(f"{t}: ⚡ SHORT flattened ${px:.2f} — {reason}")
        st.session_state.cooldown = BOS_FLAT_COOLDOWN
    st.session_state.markers.append((t, px, kind))
    st.session_state.shares = 0
    st.session_state.sl = st.session_state.tp = st.session_state.entry = None
    st.session_state.side = None

def advance(steps):
    for _ in range(steps):
        if st.session_state.step >= len(st.session_state.df) - 1:
            st.toast("Session closed.", icon="🔔")
            break
        st.session_state.step += 1
        i = st.session_state.step
        df = st.session_state.df
        c = df.iloc[i]
        t = c["label"]
        visible = df.iloc[: i + 1]
        labeled, bos, choch = detect_structure(visible)

        if st.session_state.shares != 0:
            sh, sl, tp = st.session_state.shares, st.session_state.sl, st.session_state.tp
            if sh > 0:
                if c["low"] <= sl:
                    _close_long(min(sl, float(c["open"])), t, "SL")
                elif c["high"] >= tp:
                    _close_long(max(tp, float(c["open"])), t, "TP")
                elif any(b["i"] == i and b["dir"] == "down" for b in bos):
                    _close_long(float(c["close"]), t, "FLAT", "BOS↓ against long")
            else:
                if c["high"] >= sl:
                    _close_short(max(sl, float(c["open"])), t, "SL")
                elif c["low"] <= tp:
                    _close_short(min(tp, float(c["open"])), t, "TP")
                elif any(b["i"] == i and b["dir"] == "up" for b in bos):
                    _close_short(float(c["close"]), t, "FLAT", "BOS↑ against short")
            continue

        if st.session_state.cooldown > 0:
            st.session_state.cooldown -= 1
            continue
        if i < WARMUP_BARS:
            continue

        conf = [s for s in labeled if s["i"] + SWING_K <= i]
        zones = build_zones(conf, float(visible["close"].iloc[-1]))
        decision = bot_decide(visible, i, labeled, bos, choch, zones)
        if decision:
            px = float(c["close"])
            sh = TRADE_AMT / px
            if decision["side"] == "LONG":
                st.session_state.balance -= TRADE_AMT
                st.session_state.shares = sh
            else:
                st.session_state.balance += TRADE_AMT
                st.session_state.shares = -sh
            st.session_state.entry = px
            st.session_state.sl = decision["sl"]
            st.session_state.tp = decision["tp"]
            st.session_state.side = decision["side"]
            st.session_state.log.append(
                f"{t}: 🤖 {decision['side']} ${TRADE_AMT:.0f} at ${px:.2f} "
                f"(SL {decision['sl']} / TP {decision['tp']}) — {decision['why']}"
            )
            st.session_state.markers.append((t, px, decision["side"]))

def badge(fig, x, y, text, pal, yshift=0, arrow=False):
    fig.add_annotation(
        x=x, y=y, text=f"<b>{text}</b>",
        showarrow=arrow, arrowhead=2, arrowsize=1, arrowwidth=1.4,
        arrowcolor="#111111", yshift=yshift,
        font=dict(size=11, color=pal["fg"], family="Arial"),
        bgcolor=pal["bg"], bordercolor="#111111", borderwidth=1, borderpad=3,
        opacity=1, align="center",
    )

st.title("🤖 Auto-Trader Bot — Trial v6")
st.caption(
    "Same rules every session. **Floor/ceiling must win by 2 touches** (else BOTH, not traded). "
    "**No fade** of a zone after CHoCH/BOS through it. Structural BOS still ignores pivots inside a 3+ shelf."
)

st.sidebar.header("Setup")
ticker = st.sidebar.text_input("Ticker", "QQQ").upper()
day = st.sidebar.date_input("Date", datetime.now().date() - timedelta(days=2))
show_struct = st.sidebar.checkbox("Show structure labels", True)
show_zones = st.sidebar.checkbox("Show S/R zones", True)
st.sidebar.markdown(
    f"`POLARITY_EDGE={POLARITY_EDGE}` · trade **{MIN_TRADE_TOUCHES}+** · "
    f"no short after CHoCH↑/BOS↑ through the zone"
)

raw = fetch_session(ticker, day)
if raw is None or raw.empty:
    st.sidebar.error("No 1-minute RTH data for that date (Yahoo 1m ≈ last 7 days).")
else:
    st.sidebar.success(f"{len(raw)} bars  {raw['label'].iloc[0]} → {raw['label'].iloc[-1]}")
    if st.sidebar.button("🚀 Start / Reset"):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.session_state.df = raw
        st.session_state.start_idx = 0
        st.session_state.step = min(WARMUP_BARS, len(raw) - 2)
        st.session_state.active = True
        st.rerun()

if st.session_state.active and len(st.session_state.df) > 0:
    i = st.session_state.step
    df = st.session_state.df
    vis = df.iloc[: i + 1]
    price = float(vis["close"].iloc[-1])
    labeled, bos, choch = detect_structure(vis)
    conf = [s for s in labeled if s["i"] + SWING_K <= i]
    zones = build_zones(conf, price)
    trend = trend_state(bos, choch, i, labeled)
    noise = float((vis["high"] - vis["low"]).iloc[max(0, len(vis) - 20):].mean() or 0.3)

    pos_val = st.session_state.shares * price
    equity = st.session_state.balance + pos_val

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${price:.2f}")
    c2.metric("Equity", f"${equity:.2f}", f"${equity - 1000:.2f}")
    c3.metric(
        "Position",
        st.session_state.side or "FLAT",
        f"SL {st.session_state.sl} / TP {st.session_state.tp}" if st.session_state.side else "",
    )
    c4.metric("Trend (last BOS / CHoCH)", trend)

    fig = go.Figure([go.Candlestick(
        x=vis["label"], open=vis["open"], high=vis["high"],
        low=vis["low"], close=vis["close"], name="price",
        increasing=dict(line=dict(color="#15803d"), fillcolor="#22c55e"),
        decreasing=dict(line=dict(color="#b91c1c"), fillcolor="#ef4444"),
    )])

    if show_struct:
        for s in conf:
            x = vis["label"].iloc[s["i"]]
            if s["label"] in ("HH", "HL"):
                pal, ys = C_HH, (18 if s["type"] == "H" else -18)
            elif s["label"] in ("LH", "LL"):
                pal, ys = C_LH, (18 if s["type"] == "H" else -18)
            else:
                pal, ys = C_H, (18 if s["type"] == "H" else -18)
            badge(fig, x, s["price"], s["label"], pal, yshift=ys)
        for b in bos:
            x = vis["label"].iloc[b["i"]]
            if b["dir"] == "up":
                badge(fig, x, float(vis["high"].iloc[b["i"]]), "BOS↑", C_BOS_UP, yshift=22, arrow=True)
            else:
                badge(fig, x, float(vis["low"].iloc[b["i"]]), "BOS↓", C_BOS_DN, yshift=-22, arrow=True)
        for h in choch:
            x = vis["label"].iloc[h["i"]]
            if h["dir"] == "up":
                badge(fig, x, float(vis["high"].iloc[h["i"]]), "CHoCH↑", C_CH_UP, yshift=22, arrow=True)
            else:
                badge(fig, x, float(vis["low"].iloc[h["i"]]), "CHoCH↓", C_CH_DN, yshift=-22, arrow=True)

    if show_zones:
        last_x = vis["label"].iloc[-1]
        for z in zones:
            role = zone_role(z, price, noise)
            if role == "floor":
                col, tag = C_FLOOR, "FLOOR"
            elif role == "ceiling":
                col, tag = C_CEIL, "CEIL"
            elif role == "both":
                col, tag = "#a16207", "BOTH"
            else:
                col, tag = C_BROKEN, role.replace("_", " ").upper()
            fig.add_hline(y=z["mid"], line=dict(color=col, width=min(1 + z["touches"], 4), dash="dot"))
            fig.add_annotation(
                x=last_x, y=z["mid"], xanchor="left", xref="x",
                text=(f"<b> {z['mid']:.2f} {tag} "
                      f"{z['low_touches']}L/{z['high_touches']}H</b>"),
                showarrow=False,
                font=dict(size=10, color="#ffffff", family="Arial"),
                bgcolor=col, bordercolor="#111111", borderwidth=1, borderpad=3,
            )

    if st.session_state.side:
        fig.add_hline(y=st.session_state.sl, line=dict(color="#ea580c", width=2, dash="dash"))
        fig.add_hline(y=st.session_state.tp, line=dict(color="#0369a1", width=2, dash="dash"))

    for t, p, kind in st.session_state.markers:
        sym = {"LONG": "triangle-up", "SHORT": "triangle-down",
               "TP": "star", "SL": "x", "FLAT": "diamond"}[kind]
        col = {"LONG": "#166534", "SHORT": "#991b1b",
               "TP": "#0369a1", "SL": "#ea580c", "FLAT": "#7c3aed"}[kind]
        fig.add_trace(go.Scatter(
            x=[t], y=[p], mode="markers", showlegend=False,
            marker=dict(symbol=sym, size=13, color=col, line=dict(width=1, color="#111111")),
        ))

    fig.update_layout(
        template="plotly_white", height=660, dragmode="pan",
        paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
        xaxis_rangeslider_visible=False, font=dict(color="#111111"),
        title=f"{ticker} {day} | {vis['label'].iloc[-1]} | v6 session-only | {trend}",
        margin=dict(r=160),
    )
    fig.update_xaxes(type="category", nticks=12, gridcolor="#e5e7eb", linecolor="#111111")
    fig.update_yaxes(gridcolor="#e5e7eb", linecolor="#111111")
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    a1, a2, a3, _ = st.columns([1, 1, 1, 3])
    if a1.button("▶️ +1 Min"):
        advance(1); st.rerun()
    if a2.button("⏩ +5 Min"):
        advance(5); st.rerun()
    if a3.button("⏭️ +15 Min"):
        advance(15); st.rerun()

    if st.session_state.log:
        with st.expander("📝 Bot Decision Log", expanded=True):
            for line in reversed(st.session_state.log):
                st.text(line)
else:
    st.info("Pick any recent date and press Start. Grade polarity + no-fade-after-break — not PnL.")
