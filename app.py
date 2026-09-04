# ============================================================
#  BOT TRIAL APP v2 - CURRENT SESSION ONLY
#  Structure / S/R / BOS / trades never look at prior days
# ============================================================
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="🤖 S&R Auto-Trader Bot v2")

# ==========================================
# 1. SESSION STATE
# ==========================================
defaults = {
    "active": False, "balance": 1000.0, "shares": 0.0,
    "entry": None, "sl": None, "tp": None, "side": None,
    "step": 0, "df": pd.DataFrame(), "log": [], "start_idx": 0,
    "cooldown": 0, "markers": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

TRADE_AMT = 200.0
SWING_K = 5          # 5 bars each side on 1m ≈ 10-min swing (was 8)
ZONE_TOL = 0.0012    # 0.12% cluster for S/R
MIN_RR = 1.2
COOLDOWN_BARS = 15
MIN_SWING_PCT = 0.0008   # ignore wiggles smaller than 0.08% (~$0.58 on QQQ@720)
WARMUP_BARS = 30         # no trades until the open has some structure

# ==========================================
# 2. DATA — selected calendar day, RTH only, no 7-day lookback
# ==========================================
@st.cache_data(ttl=3600)
def fetch_session(ticker, day):
    """1-minute bars for `day` only, regular hours 9:30–16:00 ET."""
    start = pd.Timestamp(day)
    end = start + timedelta(days=1)
    d = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1m",
        auto_adjust=True,
        progress=False,
        prepost=False,          # no pre/post market
    )
    if d is None or d.empty:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    df = d.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Normalise to US/Eastern so 9:30 means the cash open
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    tod = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df = df[(tod >= 9 * 60 + 30) & (tod < 16 * 60)].copy()
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    df = df.dropna(subset=["close"]).drop_duplicates(subset="timestamp")
    df = df[df["timestamp"].dt.date == day].copy()
    df["label"] = df["timestamp"].dt.strftime("%H:%M")
    return df.reset_index(drop=True)

# ==========================================
# 3. STRUCTURE ENGINE — session-local, no lookahead
# ==========================================
def detect_structure(df):
    """HH/HL/LH/LL + BOS using ONLY the bars in `df` (already today)."""
    if df is None or len(df) < SWING_K * 2 + 1:
        return [], []

    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    ref = float(df["close"].iloc[-1])
    min_size = max(ref * MIN_SWING_PCT, 1e-6)

    raw = []
    for i in range(SWING_K, n - SWING_K):
        window_h = highs[i - SWING_K: i + SWING_K + 1]
        window_l = lows[i - SWING_K: i + SWING_K + 1]
        if highs[i] == window_h.max() and (highs[i] - window_l.min()) >= min_size:
            raw.append((i, float(highs[i]), "H"))
        if lows[i] == window_l.min() and (window_h.max() - lows[i]) >= min_size:
            raw.append((i, float(lows[i]), "L"))
    raw.sort(key=lambda x: x[0])

    # Keep the more extreme of two consecutive same-type swings
    cleaned = []
    for s in raw:
        if cleaned and cleaned[-1][2] == s[2]:
            if (s[2] == "H" and s[1] >= cleaned[-1][1]) or \
               (s[2] == "L" and s[1] <= cleaned[-1][1]):
                cleaned[-1] = s
        else:
            cleaned.append(s)

    # First high/low of the DAY are just H / L — never HL vs yesterday
    labeled, last_h, last_l = [], None, None
    for idx, price, typ in cleaned:
        if typ == "H":
            lab = "H" if last_h is None else ("HH" if price > last_h else "LH")
            last_h = price
        else:
            lab = "L" if last_l is None else ("HL" if price > last_l else "LL")
            last_l = price
        labeled.append({"i": idx, "price": price, "type": typ, "label": lab})

    # BOS: close beyond the most recently *confirmed* session swing
    # A swing at bar j is confirmed only once j+SWING_K bars exist.
    bos, lh, ll_, ptr = [], None, None, 0
    for i in range(n):
        while ptr < len(labeled) and labeled[ptr]["i"] + SWING_K <= i:
            s = labeled[ptr]
            if s["type"] == "H":
                lh = s["price"]
            else:
                ll_ = s["price"]
            ptr += 1
        c = float(df["close"].iloc[i])
        if lh is not None and c > lh:
            bos.append({"i": i, "price": lh, "dir": "up"})
            lh = None
        elif ll_ is not None and c < ll_:
            bos.append({"i": i, "price": ll_, "dir": "down"})
            ll_ = None
    return labeled, bos

def trend_state(labeled, bos, i):
    """Permission comes from the last SESSION BOS; labels are fallback."""
    last_bos = None
    for b in bos:
        if b["i"] <= i:
            last_bos = b
    if last_bos is not None:
        return "BULL" if last_bos["dir"] == "up" else "BEAR"

    confirmed = [s for s in labeled if s["i"] + SWING_K <= i]
    labs = [s["label"] for s in confirmed[-4:] if s["label"] in ("HH", "HL", "LH", "LL")]
    if not labs:
        return "NEUTRAL"
    bull = labs.count("HH") + labs.count("HL")
    bear = labs.count("LH") + labs.count("LL")
    if bull >= 3:
        return "BULL"
    if bear >= 3:
        return "BEAR"
    return "NEUTRAL"

# ==========================================
# 4. S/R ZONES — today's swings, 2+ touches = concrete
# ==========================================
def build_zones(labeled, ref_price):
    zones = []
    for s in labeled:
        placed = False
        for z in zones:
            if abs(s["price"] - z["mid"]) / max(ref_price, 1e-9) < ZONE_TOL:
                z["prices"].append(s["price"])
                z["mid"] = float(np.mean(z["prices"]))
                z["touches"] += 1
                placed = True
                break
        if not placed:
            zones.append({"mid": s["price"], "prices": [s["price"]], "touches": 1})
    return [z for z in zones if z["touches"] >= 2]

# ==========================================
# 5. CANDLE PATTERNS (unchanged idea)
# ==========================================
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
        return "bull"                          # bullish engulfing
    if dn_wick > 2 * body and up_wick < body and c >= o:
        return "bull"                          # hammer
    if pc > po and c < o and c <= po and o >= pc:
        return "bear"                          # bearish engulfing
    if up_wick > 2 * body and dn_wick < body and c <= o:
        return "bear"                          # shooting star
    if i >= 2:
        b1 = df["close"].iloc[i - 1] - df["open"].iloc[i - 1]
        b2 = c - o
        avg = (df["high"] - df["low"]).iloc[max(0, i - 20):i].mean()
        if avg and b1 > 0.3 * avg and b2 > 0.3 * avg:
            return "bull"
        if avg and b1 < -0.3 * avg and b2 < -0.3 * avg:
            return "bear"
    return None

# ==========================================
# 6. BOT BRAIN
# ==========================================
def bot_decide(df, i, labeled, bos, zones):
    price = float(df["close"].iloc[i])
    noise = float((df["high"] - df["low"]).iloc[max(0, i - 20): i + 1].mean())
    trend = trend_state(labeled, bos, i)
    sig = candle_signal(df, i)

    sup = [z for z in zones if z["mid"] < price]
    res = [z for z in zones if z["mid"] > price]
    nearest_sup = max(sup, key=lambda z: z["mid"]) if sup else None
    nearest_res = min(res, key=lambda z: z["mid"]) if res else None

    # LONG: not in a bearish (last BOS down) tape, bullish candle, at a floor
    if trend != "BEAR" and sig == "bull" and nearest_sup and nearest_res:
        if (price - nearest_sup["mid"]) < 2.5 * noise:
            slp = nearest_sup["mid"] - max(2.0 * noise, price * 0.0005)
            tpp = nearest_res["mid"] - 0.5 * noise
            risk, reward = price - slp, tpp - price
            if risk > 0 and reward / risk >= MIN_RR:
                return {
                    "side": "LONG", "sl": round(slp, 2), "tp": round(tpp, 2),
                    "why": (f"{trend} | floor {nearest_sup['mid']:.2f} "
                            f"({nearest_sup['touches']}x) | RR {reward/risk:.1f}"),
                }

    # SHORT: not in a bullish (last BOS up) tape, bearish candle, at a ceiling
    if trend != "BULL" and sig == "bear" and nearest_res and nearest_sup:
        if (nearest_res["mid"] - price) < 2.5 * noise:
            slp = nearest_res["mid"] + max(2.0 * noise, price * 0.0005)
            tpp = nearest_sup["mid"] + 0.5 * noise
            risk, reward = slp - price, price - tpp
            if risk > 0 and reward / risk >= MIN_RR:
                return {
                    "side": "SHORT", "sl": round(slp, 2), "tp": round(tpp, 2),
                    "why": (f"{trend} | ceiling {nearest_res['mid']:.2f} "
                            f"({nearest_res['touches']}x) | RR {reward/risk:.1f}"),
                }
    return None

# ==========================================
# 7. TIME ADVANCE + AUTO EXECUTION
# ==========================================
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

        # --- manage open position ---
        if st.session_state.shares != 0:
            sh, sl, tp = st.session_state.shares, st.session_state.sl, st.session_state.tp
            if sh > 0:  # long
                if c["low"] <= sl:
                    px = min(sl, float(c["open"]))
                    st.session_state.balance += sh * px
                    st.session_state.log.append(f"{t}: 🛑 LONG stopped ${px:.2f}")
                    st.session_state.markers.append((t, px, "SL"))
                    st.session_state.shares = 0
                    st.session_state.cooldown = COOLDOWN_BARS
                elif c["high"] >= tp:
                    px = max(tp, float(c["open"]))
                    st.session_state.balance += sh * px
                    st.session_state.log.append(f"{t}: 🎯 LONG target ${px:.2f}")
                    st.session_state.markers.append((t, px, "TP"))
                    st.session_state.shares = 0
            else:       # short
                sh = abs(sh)
                if c["high"] >= sl:
                    px = max(sl, float(c["open"]))
                    st.session_state.balance -= sh * px
                    st.session_state.log.append(f"{t}: 🛑 SHORT stopped ${px:.2f}")
                    st.session_state.markers.append((t, px, "SL"))
                    st.session_state.shares = 0
                    st.session_state.cooldown = COOLDOWN_BARS
                elif c["low"] <= tp:
                    px = min(tp, float(c["open"]))
                    st.session_state.balance -= sh * px
                    st.session_state.log.append(f"{t}: 🎯 SHORT covered ${px:.2f}")
                    st.session_state.markers.append((t, px, "TP"))
                    st.session_state.shares = 0
            if st.session_state.shares == 0:
                st.session_state.sl = st.session_state.tp = st.session_state.entry = None
                st.session_state.side = None
            continue  # never flip the same bar as an exit

        if st.session_state.cooldown > 0:
            st.session_state.cooldown -= 1
            continue

        if i < WARMUP_BARS:
            continue

        visible = df.iloc[: i + 1]
        labeled, bos = detect_structure(visible)
        zones = build_zones(labeled, float(visible["close"].iloc[-1]))
        decision = bot_decide(visible, i, labeled, bos, zones)
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

# ==========================================
# 8. UI
# ==========================================
st.title("🤖 Auto-Trader Bot — Trial v2 (current day only)")
st.caption("Swings, BOS, S/R and trades are computed from **today’s RTH bars only**. "
           "No overnight levels, no last-week 713 ghosts.")

st.sidebar.header("Setup")
ticker = st.sidebar.text_input("Ticker", "QQQ").upper()
day = st.sidebar.date_input("Date", datetime.now().date() - timedelta(days=2))
show_struct = st.sidebar.checkbox("Show structure labels", True)
show_zones = st.sidebar.checkbox("Show S/R zones", True)

raw = fetch_session(ticker, day)
if raw is None or raw.empty:
    st.sidebar.error("No 1-minute RTH data for that date (yfinance 1m only keeps ~7 days).")
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
    labeled, bos = detect_structure(vis)
    zones = build_zones(labeled, price)
    trend = trend_state(labeled, bos, i)

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
    c4.metric("Bot trend (last session BOS)", trend)

    fig = go.Figure([go.Candlestick(
        x=vis["label"], open=vis["open"], high=vis["high"],
        low=vis["low"], close=vis["close"], name="price",
    )])

    if show_struct:
        for s in labeled:
            if s["i"] + SWING_K > i:
                continue  # unconfirmed — do not draw
            up = s["type"] == "L"
            bullish_lab = s["label"] in ("HH", "HL", "H")
            fig.add_annotation(
                x=vis["label"].iloc[s["i"]], y=s["price"],
                text=s["label"], showarrow=False,
                yshift=-16 if up else 16,
                font=dict(size=11, color="lime" if bullish_lab else "red"),
                bgcolor="rgba(0,0,0,0.45)",
            )
        for b in bos:
            fig.add_annotation(
                x=vis["label"].iloc[b["i"]], y=b["price"],
                text="BOS↑" if b["dir"] == "up" else "BOS↓",
                showarrow=True, arrowhead=2,
                font=dict(size=11, color="#111"),
                bgcolor="yellow",
            )

    if show_zones:
        last_x = vis["label"].iloc[-1]
        for z in zones:
            col = "lime" if z["mid"] < price else "red"
            fig.add_hline(
                y=z["mid"],
                line=dict(color=col, width=min(z["touches"], 3), dash="dot"),
            )
            fig.add_annotation(
                x=last_x, y=z["mid"], xanchor="left",
                text=f" {z['mid']:.2f} ({z['touches']}x)",
                showarrow=False,
                font=dict(size=10, color=col),
            )

    if st.session_state.side:
        fig.add_hline(y=st.session_state.sl, line=dict(color="orange", dash="dash"))
        fig.add_hline(y=st.session_state.tp, line=dict(color="cyan", dash="dash"))

    for t, p, kind in st.session_state.markers:
        sym = {"LONG": "triangle-up", "SHORT": "triangle-down", "TP": "star", "SL": "x"}[kind]
        col = {"LONG": "lime", "SHORT": "red", "TP": "cyan", "SL": "orange"}[kind]
        fig.add_trace(go.Scatter(
            x=[t], y=[p], mode="markers", showlegend=False,
            marker=dict(symbol=sym, size=12, color=col),
        ))

    fig.update_layout(
        template="plotly_dark", height=640, dragmode="pan",
        xaxis_rangeslider_visible=False,
        title=f"{ticker} {day} | {vis['label'].iloc[-1]} | session-only | trend {trend}",
    )
    fig.update_xaxes(type="category", nticks=12)
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
    st.info("Pick a **recent** date (1m data is only ~7 days on Yahoo) and press Start. "
            "Then step +1 / +5 / +15 and grade the labels first — trades second.")
