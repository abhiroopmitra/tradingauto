# ============================================================
#  BOT TRIAL APP v2 - Auto Structure + Auto Trading
#  Past days = analysis only. Chart = current day only.
# ============================================================
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="🤖 S&R Auto-Trader Bot")

# ---------------- CONFIG ----------------
TRADE_AMT = 200.0
SWING_K = 8            # bars each side to confirm swing
ZONE_TOL = 0.0012      # zone clustering tolerance
MIN_RR = 1.2           # minimum reward/risk
COOLDOWN_BARS = 15     # no-trade window after stop-out
MAX_ZONES_SHOWN = 3    # nearest zones each side drawn on chart

# ---------------- STATE ----------------
defaults = {
    "active": False, "balance": 1000.0, "shares": 0.0,
    "entry": None, "sl": None, "tp": None, "side": None,
    "step": 0, "df": pd.DataFrame(), "log": [], "start_idx": 0,
    "cooldown": 0, "markers": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- DATA ----------------
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
    df["label"] = df["timestamp"].dt.strftime("%H:%M")
    return df.reset_index(drop=True)

# ---------------- STRUCTURE ENGINE ----------------
def detect_structure(df):
    """Confirmed swings only (no lookahead beyond SWING_K)."""
    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    swings = []
    for i in range(SWING_K, n - SWING_K):
        if highs[i] == max(highs[i-SWING_K:i+SWING_K+1]):
            swings.append((i, highs[i], "H"))
        if lows[i] == min(lows[i-SWING_K:i+SWING_K+1]):
            swings.append((i, lows[i], "L"))
    swings.sort(key=lambda x: x[0])
    cleaned = []
    for s in swings:
        if cleaned and cleaned[-1][2] == s[2]:
            if (s[2] == "H" and s[1] >= cleaned[-1][1]) or \
               (s[2] == "L" and s[1] <= cleaned[-1][1]):
                cleaned[-1] = s
        else:
            cleaned.append(s)
    labeled, last_h, last_l = [], None, None
    for idx, price, typ in cleaned:
        if typ == "H":
            lab = "HH" if (last_h is not None and price > last_h) else "LH"
            last_h = price
        else:
            lab = "HL" if (last_l is not None and price > last_l) else "LL"
            last_l = price
        labeled.append({"i": idx, "price": price, "type": typ, "label": lab})

    # BOS detection
    bos, lh, ll_, ptr = [], None, None, 0
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
    labs = [s["label"] for s in labeled[-4:]]
    bull = labs.count("HH") + labs.count("HL")
    bear = labs.count("LH") + labs.count("LL")
    if bull >= 3: return "BULL"
    if bear >= 3: return "BEAR"
    return "NEUTRAL"

def build_zones(labeled, ref_price):
    zones = []
    for s in labeled:
        placed = False
        for z in zones:
            if abs(s["price"] - z["mid"]) / ref_price < ZONE_TOL:
                z["prices"].append(s["price"])
                z["mid"] = np.mean(z["prices"]); z["touches"] += 1
                placed = True; break
        if not placed:
            zones.append({"mid": s["price"], "prices": [s["price"]], "touches": 1})
    return [z for z in zones if z["touches"] >= 2]

# ---------------- CANDLE PATTERNS ----------------
def candle_signal(df, i):
    if i < 2: return None
    o, h, l, c = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
    po, pc = df["open"].iloc[i-1], df["close"].iloc[i-1]
    body = abs(c - o)
    up_w = h - max(o, c); dn_w = min(o, c) - l
    if pc < po and c > o and c >= po and o <= pc: return "bull"      # engulfing
    if dn_w > 2*body and up_w < body and c >= o: return "bull"       # hammer
    if pc > po and c < o and c <= po and o >= pc: return "bear"
    if up_w > 2*body and dn_w < body and c <= o: return "bear"       # shooting star
    avg = (df["high"] - df["low"]).iloc[max(0,i-20):i].mean()
    b1, b2 = pc - po, c - o
    if b1 > 0.3*avg and b2 > 0.3*avg: return "bull"                  # 2 strong greens
    if b1 < -0.3*avg and b2 < -0.3*avg: return "bear"                # 2 strong reds
    return None

# ---------------- BOT BRAIN ----------------
def bot_decide(df, i, labeled, zones):
    price = df["close"].iloc[i]
    noise = (df["high"] - df["low"]).iloc[max(0,i-20):i+1].mean()
    trend = trend_state([s for s in labeled if s["i"] + SWING_K <= i])
    sig = candle_signal(df, i)
    sup = [z for z in zones if z["mid"] < price]
    res = [z for z in zones if z["mid"] > price]
    ns = max(sup, key=lambda z: z["mid"]) if sup else None
    nr = min(res, key=lambda z: z["mid"]) if res else None

    if trend != "BEAR" and sig == "bull" and ns and nr:
        if (price - ns["mid"]) < 2.5 * noise:                        # near floor
            slp = ns["mid"] - max(1.5*noise, price*0.0005)           # beyond structure
            tpp = nr["mid"] - 0.5*noise                              # nearest shelf
            risk, rew = price - slp, tpp - price
            if risk > 0 and rew/risk >= MIN_RR:
                return {"side":"LONG","sl":round(slp,2),"tp":round(tpp,2),
                        "why":f"{trend} | floor {ns['mid']:.2f} ({ns['touches']}x) | RR {rew/risk:.1f}"}

    if trend != "BULL" and sig == "bear" and nr and ns:
        if (nr["mid"] - price) < 2.5 * noise:                        # near ceiling
            slp = nr["mid"] + max(1.5*noise, price*0.0005)
            tpp = ns["mid"] + 0.5*noise
            risk, rew = slp - price, price - tpp
            if risk > 0 and rew/risk >= MIN_RR:
                return {"side":"SHORT","sl":round(slp,2),"tp":round(tpp,2),
                        "why":f"{trend} | ceiling {nr['mid']:.2f} ({nr['touches']}x) | RR {rew/risk:.1f}"}
    return None

# ---------------- EXECUTION ENGINE ----------------
def advance(steps):
    for _ in range(steps):
        if st.session_state.step >= len(st.session_state.df) - 1:
            st.toast("Market closed!", icon="🔔"); break
        st.session_state.step += 1
        i = st.session_state.step
        df = st.session_state.df
        c = df.iloc[i]; t = c["label"]

        # manage open position
        if st.session_state.shares != 0:
            sh = st.session_state.shares
            sl, tp = st.session_state.sl, st.session_state.tp
            if sh > 0:
                if c["low"] <= sl:
                    px = min(sl, c["open"])
                    st.session_state.balance += sh*px
                    st.session_state.log.append(f"{t}: 🛑 LONG stopped ${px:.2f}")
                    st.session_state.markers.append((t, px, "SL"))
                    st.session_state.shares = 0; st.session_state.cooldown = COOLDOWN_BARS
                elif c["high"] >= tp:
                    px = max(tp, c["open"])
                    st.session_state.balance += sh*px
                    st.session_state.log.append(f"{t}: 🎯 LONG target ${px:.2f}")
                    st.session_state.markers.append((t, px, "TP"))
                    st.session_state.shares = 0
            else:
                sh = abs(sh)
                if c["high"] >= sl:
                    px = max(sl, c["open"])
                    st.session_state.balance -= sh*px
                    st.session_state.log.append(f"{t}: 🛑 SHORT stopped ${px:.2f}")
                    st.session_state.markers.append((t, px, "SL"))
                    st.session_state.shares = 0; st.session_state.cooldown = COOLDOWN_BARS
                elif c["low"] <= tp:
                    px = min(tp, c["open"])
                    st.session_state.balance -= sh*px
                    st.session_state.log.append(f"{t}: 🎯 SHORT covered ${px:.2f}")
                    st.session_state.markers.append((t, px, "TP"))
                    st.session_state.shares = 0
            if st.session_state.shares == 0:
                st.session_state.sl = st.session_state.tp = st.session_state.entry = None
                st.session_state.side = None
            continue

        if st.session_state.cooldown > 0:
            st.session_state.cooldown -= 1; continue
        if i < st.session_state.start_idx:
            continue

        visible = df.iloc[:i+1]        # includes past days = bot's context
        labeled, _ = detect_structure(visible)
        zones = build_zones(labeled, visible["close"].iloc[-1])
        d = bot_decide(visible, i, labeled, zones)
        if d:
            px = c["close"]; sh = TRADE_AMT/px
            if d["side"] == "LONG":
                st.session_state.balance -= TRADE_AMT; st.session_state.shares = sh
            else:
                st.session_state.balance += TRADE_AMT; st.session_state.shares = -sh
            st.session_state.entry, st.session_state.sl = px, d["sl"]
            st.session_state.tp, st.session_state.side = d["tp"], d["side"]
            st.session_state.log.append(
                f"{t}: 🤖 {d['side']} at ${px:.2f} (SL {d['sl']} / TP {d['tp']}) — {d['why']}")
            st.session_state.markers.append((t, px, d["side"]))

# ---------------- UI ----------------
st.title("🤖 Auto-Trader Bot v2")
st.sidebar.header("Setup")
ticker = st.sidebar.text_input("Ticker", "QQQ").upper()
day = st.sidebar.date_input("Date", datetime.now().date() - timedelta(days=2))

raw = fetch(ticker, day)
if raw is not None and day in raw["date_only"].values:
    mask = raw["date_only"] == day
    if st.sidebar.button("🚀 Start / Reset"):
        st.session_state.update({k: v for k, v in defaults.items()
                                 if k not in ("df","start_idx","step")})
        st.session_state.df = raw
        st.session_state.start_idx = raw.index[mask][0]
        st.session_state.step = raw.index[mask][0] + 15
        st.session_state.active = True
        st.rerun()
else:
    st.sidebar.error("No 1m data for that date.")

show_struct = st.sidebar.checkbox("Structure labels", True)
show_zones = st.sidebar.checkbox("S/R zones", True)

if st.session_state.active:
    i = st.session_state.step
    df = st.session_state.df
    full_vis = df.iloc[:i+1]                                  # bot's full context
    day_vis = full_vis.iloc[st.session_state.start_idx - full_vis.index[0]:] \
              if False else full_vis[full_vis.index >= st.session_state.start_idx]  # chart = today only
    price = full_vis["close"].iloc[-1]

    labeled, bos = detect_structure(full_vis)
    zones = build_zones(labeled, price)
    trend = trend_state([s for s in labeled if s["i"] + SWING_K <= i])

    pos_val = st.session_state.shares * price
    equity = st.session_state.balance + pos_val
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${price:.2f}")
    c2.metric("Equity", f"${equity:.2f}", f"${equity-1000:.2f}")
    c3.metric("Position", st.session_state.side or "FLAT",
              f"SL {st.session_state.sl} / TP {st.session_state.tp}" if st.session_state.side else "")
    c4.metric("Bot Trend Read", trend)

    # ---- CHART: current day only, white theme ----
    fig = go.Figure([go.Candlestick(
        x=day_vis["label"], open=day_vis["open"], high=day_vis["high"],
        low=day_vis["low"], close=day_vis["close"],
        increasing_line_color="#089981", decreasing_line_color="#e02f2f")])

    day_first_idx = day_vis.index[0]

    if show_struct:
        for s in labeled:
            if s["i"] + SWING_K > i: continue
            if s["i"] < day_first_idx: continue               # only label today's swings
            bullish = s["label"] in ("HH", "HL")
            fig.add_annotation(
                x=df["label"].iloc[s["i"]], y=s["price"], text=f"<b>{s['label']}</b>",
                showarrow=False, yshift=-16 if s["type"]=="L" else 16,
                font=dict(size=12, color="green" if bullish else "red"),
                bgcolor="white", bordercolor="black", borderwidth=1)
        for b in bos:
            if b["i"] < day_first_idx: continue
            fig.add_annotation(
                x=df["label"].iloc[b["i"]], y=b["price"],
                text=f"<b>BOS{'↑' if b['dir']=='up' else '↓'}</b>",
                showarrow=True, arrowhead=2, arrowcolor="black",
                font=dict(size=12, color="black"), bgcolor="yellow")

    if show_zones:
        # only nearest 3 each side = clean chart
        sup = sorted([z for z in zones if z["mid"] < price],
                     key=lambda z: -z["mid"])[:MAX_ZONES_SHOWN]
        res = sorted([z for z in zones if z["mid"] > price],
                     key=lambda z: z["mid"])[:MAX_ZONES_SHOWN]
        for z in sup:
            fig.add_hline(y=z["mid"], line=dict(color="green", width=1.5, dash="dot"),
                          annotation_text=f"S {z['mid']:.2f} ({z['touches']}x)",
                          annotation_font_color="green")
        for z in res:
            fig.add_hline(y=z["mid"], line=dict(color="red", width=1.5, dash="dot"),
                          annotation_text=f"R {z['mid']:.2f} ({z['touches']}x)",
                          annotation_font_color="red")

    if st.session_state.side:
        fig.add_hline(y=st.session_state.sl, line=dict(color="orange", width=2, dash="dash"),
                      annotation_text="STOP", annotation_font_color="orange")
        fig.add_hline(y=st.session_state.tp, line=dict(color="blue", width=2, dash="dash"),
                      annotation_text="TARGET", annotation_font_color="blue")

    day_labels = set(day_vis["label"])
    for t, p, kind in st.session_state.markers:
        if t not in day_labels: continue
        sym = {"LONG":"triangle-up","SHORT":"triangle-down","TP":"star","SL":"x"}[kind]
        col = {"LONG":"green","SHORT":"red","TP":"blue","SL":"orange"}[kind]
        fig.add_trace(go.Scatter(x=[t], y=[p], mode="markers", showlegend=False,
                                 marker=dict(symbol=sym, size=14, color=col,
                                             line=dict(width=1, color="black"))))

    fig.update_layout(template="plotly_white", height=620, dragmode="pan",
                      xaxis_rangeslider_visible=False,
                      title=f"{ticker} {day} | {day_vis['label'].iloc[-1]} | Trend: {trend}")
    fig.update_xaxes(type="category", nticks=10)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    a1, a2, a3, _ = st.columns([1,1,1,3])
    if a1.button("▶️ +1 Min"):  advance(1);  st.rerun()
    if a2.button("⏩ +5 Min"):  advance(5);  st.rerun()
    if a3.button("⏭️ +15 Min"): advance(15); st.rerun()

    if st.session_state.log:
        with st.expander("📝 Bot Decision Log", expanded=True):
            for l in reversed(st.session_state.log): st.text(l)
else:
    st.info("👈 Pick date → Start. Then click time buttons and watch the bot trade.")
