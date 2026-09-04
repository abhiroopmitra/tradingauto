# ============================================================
#  BOT TRIAL APP - Auto Structure Labeling + Auto Trading
#  (Separate from the manual Market Replay Simulator)
# ============================================================
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="🤖 S&R Auto-Trader Bot")

# ==========================================
# 1. SESSION STATE
# ==========================================
defaults = {
    "active": False, "balance": 1000.0, "shares": 0.0,
    "entry": None, "sl": None, "tp": None, "side": None,
    "step": 0, "df": pd.DataFrame(), "log": [], "start_idx": 0,
    "cooldown": 0, "markers": [],   # trade markers for chart
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

TRADE_AMT = 200.0
SWING_K = 8          # candles each side to confirm a swing (no lookahead)
ZONE_TOL = 0.0012    # 0.12% clustering tolerance for S/R zones
MIN_RR = 1.2         # reject trades below this reward/risk
COOLDOWN_BARS = 15   # no-trade window after a stop-out

# ==========================================
# 2. DATA
# ==========================================
@st.cache_data(ttl=3600)
def fetch(ticker, day):
    start = day - timedelta(days=7)
    end = day + timedelta(days=1)
    d = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"), interval="1m",
                    auto_adjust=True, progress=False)
    if d.empty: return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    df = d.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.dropna(subset=["close"]).drop_duplicates(subset="timestamp")
    df["date_only"] = df["timestamp"].dt.date
    df["label"] = df["timestamp"].dt.strftime("%m-%d %H:%M")
    return df.reset_index(drop=True)

# ==========================================
# 3. STRUCTURE ENGINE (HH / HL / LH / LL / BOS)
#    Uses ONLY data up to current step - no lookahead.
#    A swing at bar i is only "confirmed" once i+SWING_K bars exist.
# ==========================================
def detect_structure(df):
    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    swings = []  # (index, price, "H"/"L")
    for i in range(SWING_K, n - SWING_K):
        if highs[i] == max(highs[i-SWING_K:i+SWING_K+1]):
            swings.append((i, highs[i], "H"))
        if lows[i] == min(lows[i-SWING_K:i+SWING_K+1]):
            swings.append((i, lows[i], "L"))
    swings.sort(key=lambda x: x[0])

    # collapse consecutive same-type swings (keep the more extreme one)
    cleaned = []
    for s in swings:
        if cleaned and cleaned[-1][2] == s[2]:
            if (s[2] == "H" and s[1] >= cleaned[-1][1]) or \
               (s[2] == "L" and s[1] <= cleaned[-1][1]):
                cleaned[-1] = s
        else:
            cleaned.append(s)

    # label HH/LH/HL/LL
    labeled, last_h, last_l = [], None, None
    for idx, price, typ in cleaned:
        if typ == "H":
            lab = "HH" if (last_h is not None and price > last_h) else "LH"
            last_h = price
        else:
            lab = "HL" if (last_l is not None and price > last_l) else "LL"
            last_l = price
        labeled.append({"i": idx, "price": price, "type": typ, "label": lab})

    # BOS: close crosses the most recent confirmed swing of opposite side
    bos = []
    lh, ll_ = None, None
    ptr = 0
    for i in range(n):
        while ptr < len(labeled) and labeled[ptr]["i"] + SWING_K <= i:
            s = labeled[ptr]
            if s["type"] == "H": lh = s["price"]
            else: ll_ = s["price"]
            ptr += 1
        c = df["close"].iloc[i]
        if lh is not None and c > lh:
            bos.append({"i": i, "price": lh, "dir": "up"}); lh = None
        if ll_ is not None and c < ll_:
            bos.append({"i": i, "price": ll_, "dir": "down"}); ll_ = None
    return labeled, bos

def trend_state(labeled):
    """Bullish / Bearish / Neutral from the last 4 swing labels."""
    labs = [s["label"] for s in labeled[-4:]]
    bull = labs.count("HH") + labs.count("HL")
    bear = labs.count("LH") + labs.count("LL")
    if bull >= 3: return "BULL"
    if bear >= 3: return "BEAR"
    return "NEUTRAL"

# ==========================================
# 4. S/R ZONES (clustered swing prices, 2+ touches = concrete)
# ==========================================
def build_zones(labeled, ref_price):
    zones = []
    for s in labeled:
        placed = False
        for z in zones:
            if abs(s["price"] - z["mid"]) / ref_price < ZONE_TOL:
                z["prices"].append(s["price"])
                z["mid"] = np.mean(z["prices"])
                z["touches"] += 1
                placed = True
                break
        if not placed:
            zones.append({"mid": s["price"], "prices": [s["price"]], "touches": 1})
    return [z for z in zones if z["touches"] >= 2]   # concrete floors only

# ==========================================
# 5. CANDLE PATTERNS
# ==========================================
def candle_signal(df, i):
    """Returns 'bull', 'bear', or None for bar i."""
    if i < 1: return None
    o, h, l, c = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
    po, pc = df["open"].iloc[i-1], df["close"].iloc[i-1]
    rng = max(h - l, 1e-9); body = abs(c - o)
    up_wick = h - max(o, c); dn_wick = min(o, c) - l

    # Bullish engulfing / hammer
    if pc < po and c > o and c >= po and o <= pc: return "bull"
    if dn_wick > 2 * body and up_wick < body and c >= o: return "bull"     # hammer
    # Bearish engulfing / shooting star
    if pc > po and c < o and c <= po and o >= pc: return "bear"
    if up_wick > 2 * body and dn_wick < body and c <= o: return "bear"     # shooting star
    # Two strong same-color candles
    if i >= 2:
        b1 = df["close"].iloc[i-1] - df["open"].iloc[i-1]
        b2 = c - o
        avg = (df["high"] - df["low"]).iloc[max(0, i-20):i].mean()
        if b1 > 0.3*avg and b2 > 0.3*avg: return "bull"
        if b1 < -0.3*avg and b2 < -0.3*avg: return "bear"
    return None

# ==========================================
# 6. THE BOT BRAIN - runs on each new candle
# ==========================================
def bot_decide(df, i, labeled, zones):
    """All your session rules, in order. Returns a trade dict or None."""
    price = df["close"].iloc[i]
    noise = (df["high"] - df["low"]).iloc[max(0, i-20):i+1].mean()  # avg 1m range
    trend = trend_state([s for s in labeled if s["i"] + SWING_K <= i])
    sig = candle_signal(df, i)

    sup = [z for z in zones if z["mid"] < price]
    res = [z for z in zones if z["mid"] > price]
    nearest_sup = max(sup, key=lambda z: z["mid"]) if sup else None
    nearest_res = min(res, key=lambda z: z["mid"]) if res else None

    # ---- LONG: trend not bearish + at support + bullish candle ----
    if trend != "BEAR" and sig == "bull" and nearest_sup and nearest_res:
        near_floor = (price - nearest_sup["mid"]) < 2.5 * noise
        if near_floor:
            slp = nearest_sup["mid"] - max(1.5 * noise, price * 0.0005)   # beyond structure
            tpp = nearest_res["mid"] - 0.5 * noise                        # nearest shelf
            risk, reward = price - slp, tpp - price
            if risk > 0 and reward / risk >= MIN_RR:
                return {"side": "LONG", "sl": round(slp, 2), "tp": round(tpp, 2),
                        "why": f"{trend} trend | bounce at zone {nearest_sup['mid']:.2f} "
                               f"({nearest_sup['touches']} touches) | RR {reward/risk:.1f}"}

    # ---- SHORT: trend not bullish + at resistance + bearish candle ----
    if trend != "BULL" and sig == "bear" and nearest_res and nearest_sup:
        near_ceil = (nearest_res["mid"] - price) < 2.5 * noise
        if near_ceil:
            slp = nearest_res["mid"] + max(1.5 * noise, price * 0.0005)
            tpp = nearest_sup["mid"] + 0.5 * noise
            risk, reward = slp - price, price - tpp
            if risk > 0 and reward / risk >= MIN_RR:
                return {"side": "SHORT", "sl": round(slp, 2), "tp": round(tpp, 2),
                        "why": f"{trend} trend | rejection at zone {nearest_res['mid']:.2f} "
                               f"({nearest_res['touches']} touches) | RR {reward/risk:.1f}"}
    return None

# ==========================================
# 7. TIME ADVANCE + AUTO EXECUTION
# ==========================================
def advance(steps):
    for _ in range(steps):
        if st.session_state.step >= len(st.session_state.df) - 1:
            st.toast("Market closed!", icon="🔔"); break
        st.session_state.step += 1
        i = st.session_state.step
        df = st.session_state.df
        c = df.iloc[i]; t = c["label"]

        # --- manage open position ---
        if st.session_state.shares != 0:
            sh, sl, tp = st.session_state.shares, st.session_state.sl, st.session_state.tp
            if sh > 0:   # long
                if c["low"] <= sl:
                    px = min(sl, c["open"])
                    st.session_state.balance += sh * px
                    st.session_state.log.append(f"{t}: 🛑 LONG stopped at ${px:.2f}")
                    st.session_state.markers.append((t, px, "SL"))
                    st.session_state.shares = 0; st.session_state.cooldown = COOLDOWN_BARS
                elif c["high"] >= tp:
                    px = max(tp, c["open"])
                    st.session_state.balance += sh * px
                    st.session_state.log.append(f"{t}: 🎯 LONG target hit at ${px:.2f}")
                    st.session_state.markers.append((t, px, "TP"))
                    st.session_state.shares = 0
            else:        # short
                sh = abs(sh)
                if c["high"] >= sl:
                    px = max(sl, c["open"])
                    st.session_state.balance -= sh * px
                    st.session_state.log.append(f"{t}: 🛑 SHORT stopped at ${px:.2f}")
                    st.session_state.markers.append((t, px, "SL"))
                    st.session_state.shares = 0; st.session_state.cooldown = COOLDOWN_BARS
                elif c["low"] <= tp:
                    px = min(tp, c["open"])
                    st.session_state.balance -= sh * px
                    st.session_state.log.append(f"{t}: 🎯 SHORT covered at ${px:.2f}")
                    st.session_state.markers.append((t, px, "TP"))
                    st.session_state.shares = 0
            if st.session_state.shares == 0:
                st.session_state.sl = st.session_state.tp = st.session_state.entry = None
                st.session_state.side = None
            continue   # never open a new trade same bar as a close

        # --- cooldown after a stop (no revenge trading) ---
        if st.session_state.cooldown > 0:
            st.session_state.cooldown -= 1
            continue

        # --- look for a new trade (only on the sim day, only with warmup) ---
        if i < st.session_state.start_idx or i < 40:
            continue
        visible = df.iloc[:i+1]
        labeled, _ = detect_structure(visible)
        zones = build_zones(labeled, visible["close"].iloc[-1])
        decision = bot_decide(visible, i, labeled, zones)
        if decision:
            px = c["close"]; sh = TRADE_AMT / px
            if decision["side"] == "LONG":
                st.session_state.balance -= TRADE_AMT
                st.session_state.shares = sh
            else:
                st.session_state.balance += TRADE_AMT
                st.session_state.shares = -sh
            st.session_state.entry, st.session_state.sl = px, decision["sl"]
            st.session_state.tp, st.session_state.side = decision["tp"], decision["side"]
            st.session_state.log.append(
                f"{t}: 🤖 {decision['side']} ${TRADE_AMT:.0f} at ${px:.2f} "
                f"(SL {decision['sl']} / TP {decision['tp']}) — {decision['why']}")
            st.session_state.markers.append((t, px, decision["side"]))

# ==========================================
# 8. UI
# ==========================================
st.title("🤖 Auto-Trader Bot (Trial)")
st.sidebar.header("Setup")
ticker = st.sidebar.text_input("Ticker", "QQQ").upper()
day = st.sidebar.date_input("Date", datetime.now().date() - timedelta(days=2))

raw = fetch(ticker, day)
if raw is not None and day in raw["date_only"].values:
    mask = raw["date_only"] == day
    if st.sidebar.button("🚀 Start / Reset"):
        st.session_state.df = raw
        st.session_state.start_idx = raw.index[mask][0]
        st.session_state.step = raw.index[mask][0] + 30   # let 30 bars form first
        for k, v in defaults.items():
            if k not in ("df", "start_idx", "step"): st.session_state[k] = v
        st.session_state.df = raw
        st.session_state.start_idx = raw.index[mask][0]
        st.session_state.step = raw.index[mask][0] + 30
        st.session_state.active = True
        st.rerun()
else:
    st.sidebar.error("No 1m data for that date.")

show_struct = st.sidebar.checkbox("Show structure labels", True)
show_zones = st.sidebar.checkbox("Show S/R zones", True)

if st.session_state.active:
    i = st.session_state.step
    df = st.session_state.df
    vis = df.iloc[:i+1]
    price = vis["close"].iloc[-1]

    pos_val = st.session_state.shares * price
    equity = st.session_state.balance + pos_val
    labeled, bos = detect_structure(vis)
    zones = build_zones(labeled, price)
    trend = trend_state([s for s in labeled if s["i"] + SWING_K <= i])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${price:.2f}")
    c2.metric("Equity", f"${equity:.2f}", f"${equity-1000:.2f}")
    c3.metric("Position", st.session_state.side or "FLAT",
              f"SL {st.session_state.sl} / TP {st.session_state.tp}" if st.session_state.side else "")
    c4.metric("Bot Trend Read", trend)

    fig = go.Figure([go.Candlestick(x=vis["label"], open=vis["open"], high=vis["high"],
                                    low=vis["low"], close=vis["close"])])
    # structure labels (only confirmed swings)
    if show_struct:
        for s in labeled:
            if s["i"] + SWING_K > i: continue
            up = s["type"] == "L"
            fig.add_annotation(x=vis["label"].iloc[s["i"]], y=s["price"],
                               text=s["label"], showarrow=False,
                               yshift=-14 if up else 14,
                               font=dict(size=10, color="lime" if s["label"] in ("HH","HL") else "red"))
        for b in bos:
            fig.add_annotation(x=vis["label"].iloc[b["i"]], y=b["price"],
                               text="BOS↑" if b["dir"]=="up" else "BOS↓", showarrow=True,
                               arrowhead=2, font=dict(size=11, color="yellow"))
    if show_zones:
        for z in zones:
            fig.add_hline(y=z["mid"], line=dict(
                color="lime" if z["mid"] < price else "red",
                width=min(z["touches"], 3), dash="dot"))
    if st.session_state.side:
        fig.add_hline(y=st.session_state.sl, line=dict(color="orange", dash="dash"))
        fig.add_hline(y=st.session_state.tp, line=dict(color="cyan", dash="dash"))
    for t, p, kind in st.session_state.markers:
        sym = {"LONG":"triangle-up","SHORT":"triangle-down","TP":"star","SL":"x"}[kind]
        col = {"LONG":"lime","SHORT":"red","TP":"cyan","SL":"orange"}[kind]
        fig.add_trace(go.Scatter(x=[t], y=[p], mode="markers", showlegend=False,
                                 marker=dict(symbol=sym, size=12, color=col)))
    fig.update_layout(template="plotly_dark", height=620, dragmode="pan",
                      xaxis_rangeslider_visible=False,
                      title=f"{ticker} | {vis['label'].iloc[-1]} | Bot trend: {trend}")
    fig.update_xaxes(type="category", nticks=12)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    a1, a2, a3, _ = st.columns([1,1,1,3])
    if a1.button("▶️ +1 Min"):  advance(1);  st.rerun()
    if a2.button("⏩ +5 Min"):  advance(5);  st.rerun()
    if a3.button("⏭️ +15 Min"): advance(15); st.rerun()

    if st.session_state.log:
        with st.expander("📝 Bot Decision Log", expanded=True):
            for l in reversed(st.session_state.log): st.text(l)
else:
    st.info("👈 Pick a date and press Start. Then advance time and watch the bot think.")
