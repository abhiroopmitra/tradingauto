# ============================================================
# BOT V5 - GENERIC INTRADAY - NOT TUNED TO ANY DAY
# ============================================================
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta, time

st.set_page_config(layout="wide", page_title="Bot V5 Generic")
TRADE_AMT, SWING_K, MIN_RR, COOLDOWN = 200.0, 7, 1.5, 15

@st.cache_data(ttl=60)
def fetch_day(ticker, day):
    d = yf.download(ticker, start=day.strftime("%Y-%m-%d"), 
                    end=(day+timedelta(days=1)).strftime("%Y-%m-%d"),
                    interval="1m", auto_adjust=True, progress=False, prepost=False)
    if d.empty: return None
    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
    df = d.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df.rename(columns={df.columns[0]:"timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df[(df["timestamp"].dt.date==day) & 
            (df["timestamp"].dt.time>=time(9,30)) & 
            (df["timestamp"].dt.time<=time(16,0))]
    df = df.dropna(subset=["close"]).sort_values("timestamp")
    df["label"] = df["timestamp"].dt.strftime("%H:%M")
    return df.reset_index(drop=True)

def get_swings(df, k=SWING_K):
    swings=[]; h,l = df["high"].values, df["low"].values
    for i in range(k, len(df)-k):
        if h[i] == np.max(h[i-k:i+k+1]): swings.append((i, h[i], "H"))
        if l[i] == np.min(l[i-k:i+k+1]): swings.append((i, l[i], "L"))
    swings.sort(key=lambda x:x[0])
    # keep most extreme if same type consecutive, enforce alternation
    tmp=[]
    for s in swings:
        if tmp and tmp[-1][2]==s[2]:
            if s[2]=="H" and s[1]>tmp[-1][1]: tmp[-1]=s
            if s[2]=="L" and s[1]<tmp[-1][1]: tmp[-1]=s
        else: tmp.append(s)
    clean=[]
    for s in tmp:
        if not clean or clean[-1][2]!=s[2]: clean.append(s)
    return clean

def label_swings(swings):
    out=[]; last_h=None; last_l=None
    for i,p,t in swings:
        if t=="H":
            lab = "H" if last_h is None else ("HH" if p>last_h else "LH")
            last_h=p
        else:
            lab = "L" if last_l is None else ("HL" if p>last_l else "LL")
            last_l=p
        out.append({"i":i,"price":p,"type":t,"label":lab})
    return out

def detect_bos(df, labeled):
    bos=[]
    for i in range(1,len(df)):
        conf = [s for s in labeled if s["i"]+SWING_K <= i]
        if not conf: continue
        last_h = [s for s in conf if s["type"]=="H"]
        last_l = [s for s in conf if s["type"]=="L"]
        if last_h:
            lh = last_h[-1]
            if df["close"].iloc[i] > lh["price"] and df["close"].iloc[i-1] <= lh["price"]:
                bos.append({"i":i,"broken_price":lh["price"],"dir":"up","broken_idx":lh["i"]})
        if last_l:
            ll = last_l[-1]
            if df["close"].iloc[i] < ll["price"] and df["close"].iloc[i-1] >= ll["price"]:
                bos.append({"i":i,"broken_price":ll["price"],"dir":"down","broken_idx":ll["i"]})
    return bos

def build_sr(labeled, ref_price, atr):
    tol = max(0.0012, 0.4*atr/ref_price)
    sup,res=[],[]
    for s in labeled:
        bucket = sup if s["type"]=="L" else res
        placed=False
        for z in bucket:
            if abs(s["price"]-z["mid"])/ref_price < tol:
                z["prices"].append(s["price"]); z["mid"]=np.mean(z["prices"]); z["touches"]+=1; placed=True; break
        if not placed:
            bucket.append({"mid":s["price"],"prices":[s["price"]],"touches":1})
    sup = [z for z in sup if z["touches"]>=2]
    res = [z for z in res if z["touches"]>=2]
    return sup,res

def candle_signal(df,i):
    if i<1: return None, None
    o,h,l,c = df["open"].iloc[i],df["high"].iloc[i],df["low"].iloc[i],df["close"].iloc[i]
    po,pc = df["open"].iloc[i-1],df["close"].iloc[i-1]
    body=abs(c-o); uw=h-max(o,c); dw=min(o,c)-l
    if pc<po and c>o and c>=po and o<=pc: return "bull","Bull Engulf"
    if pc>po and c<o and c<=po and o>=pc: return "bear","Bear Engulf"
    if dw>2.2*body and uw<0.6*body and c>o: return "bull","Hammer"
    if uw>2.2*body and dw<0.6*body and c<o: return "bear","Shooting Star"
    return None, None

def bot_decide(df,i,labeled,sup,res,bos,atr):
    price=df["close"].iloc[i]
    sig, pat = candle_signal(df,i)
    if not sig: return None
    
    # nearest zones
    near_sup = min(sup, key=lambda z: abs(z["mid"]-price)) if sup else None
    near_res = min(res, key=lambda z: abs(z["mid"]-price)) if res else None
    
    at_sup = near_sup and abs(price-near_sup["mid"]) < 2.5*atr
    at_res = near_res and abs(price-near_res["mid"]) < 2.5*atr

    # BOS retest levels
    last_bos_up = [b for b in bos if b["dir"]=="up"][-1] if any(b["dir"]=="up" for b in bos) else None
    last_bos_down = [b for b in bos if b["dir"]=="down"][-1] if any(b["dir"]=="down" for b in bos) else None
    
    at_bos_sup = last_bos_up and abs(price-last_bos_up["broken_price"]) < 2.5*atr and price>last_bos_up["broken_price"]-atr
    at_bos_res = last_bos_down and abs(price-last_bos_down["broken_price"]) < 2.5*atr and price<last_bos_down["broken_price"]+atr

    # LONG LOGIC: at support or retest of broken resistance + bull candle
    if sig=="bull" and (at_sup or at_bos_sup):
        use_sup = near_sup if at_sup else {"mid":last_bos_up["broken_price"],"touches":1}
        sl = use_sup["mid"] - 1.5*atr
        tp = near_res["mid"]-0.5*atr if near_res else price + 3*atr
        rr = (tp-price)/(price-sl) if price>sl else 0
        if rr>=MIN_RR:
            why = f"{pat} at SUP {use_sup['mid']:.2f}({use_sup.get('touches',1)}x)" if at_sup else f"{pat} retest BOS↑ {use_sup['mid']:.2f}"
            if near_sup and near_sup["touches"]>=3: why += " [DoubleBottom]"
            return {"side":"LONG","sl":round(sl,2),"tp":round(tp,2),"why":f"{why} | RR {rr:.1f}"}

    # SHORT LOGIC: at resistance or retest of broken support + bear candle
    if sig=="bear" and (at_res or at_bos_res):
        use_res = near_res if at_res else {"mid":last_bos_down["broken_price"],"touches":1}
        sl = use_res["mid"] + 1.5*atr
        tp = near_sup["mid"]+0.5*atr if near_sup else price - 3*atr
        rr = (price-tp)/(sl-price) if sl>price else 0
        if rr>=MIN_RR:
            why = f"{pat} at RES {use_res['mid']:.2f}({use_res.get('touches',1)}x)" if at_res else f"{pat} retest BOS↓ {use_res['mid']:.2f}"
            if near_res and near_res["touches"]>=3: why += " [DoubleTop]"
            return {"side":"SHORT","sl":round(sl,2),"tp":round(tp,2),"why":f"{why} | RR {rr:.1f}"}
    return None

# --- APP STATE ---
for k,v in {"active":False,"balance":1000.0,"shares":0.0,"sl":None,"tp":None,"side":None,"step":0,"day_df":pd.DataFrame(),"log":[],"markers":[],"cooldown":0}.items():
    if k not in st.session_state: st.session_state[k]=v

def advance(steps):
    for _ in range(steps):
        if st.session_state.step>=len(st.session_state.day_df)-1: break
        st.session_state.step+=1
        i=st.session_state.step; df=st.session_state.day_df; c=df.iloc[i]
        atr = (df["high"]-df["low"]).iloc[max(0,i-20):i+1].mean()
        # manage open
        if st.session_state.shares!=0:
            sh,sl,tp=st.session_state.shares,st.session_state.sl,st.session_state.tp
            if sh>0:
                if c["low"]<=sl:
                    px=min(sl,c["open"]); st.session_state.balance+=sh*px
                    st.session_state.log.append(f"{c['label']}: 🛑 LONG stop {px:.2f}"); st.session_state.markers.append((c['label'],px,"SL")); st.session_state.shares=0; st.session_state.cooldown=COOLDOWN
                elif c["high"]>=tp:
                    px=max(tp,c["open"]); st.session_state.balance+=sh*px
                    st.session_state.log.append(f"{c['label']}: 🎯 LONG tp {px:.2f}"); st.session_state.markers.append((c['label'],px,"TP")); st.session_state.shares=0
            else:
                sh=abs(sh)
                if c["high"]>=sl:
                    px=max(sl,c["open"]); st.session_state.balance-=sh*px
                    st.session_state.log.append(f"{c['label']}: 🛑 SHORT stop {px:.2f}"); st.session_state.markers.append((c['label'],px,"SL")); st.session_state.shares=0; st.session_state.cooldown=COOLDOWN
                elif c["low"]<=tp:
                    px=min(tp,c["open"]); st.session_state.balance-=sh*px
                    st.session_state.log.append(f"{c['label']}: 🎯 SHORT cover {px:.2f}"); st.session_state.markers.append((c['label'],px,"TP")); st.session_state.shares=0
            if st.session_state.shares==0: st.session_state.sl=st.session_state.tp=st.session_state.side=None
            continue
        if st.session_state.cooldown>0: st.session_state.cooldown-=1; continue
        if i<40: continue
        vis=df.iloc[:i+1]
        swings=get_swings(vis); labeled=label_swings([s for s in swings if s[0]+SWING_K<=i])
        sup,res=build_sr(labeled, vis["close"].iloc[-1], atr)
        bos=detect_bos(vis,labeled)
        dec=bot_decide(vis,i,labeled,sup,res,bos,atr)
        if dec:
            px=c["close"]; sh=TRADE_AMT/px
            if dec["side"]=="LONG": st.session_state.balance-=TRADE_AMT; st.session_state.shares=sh
            else: st.session_state.balance+=TRADE_AMT; st.session_state.shares=-sh
            st.session_state.sl,st.session_state.tp,st.session_state.side=dec["sl"],dec["tp"],dec["side"]
            st.session_state.log.append(f"{c['label']}: 🤖 {dec['side']} ${TRADE_AMT:.0f} at {px:.2f} SL {dec['sl']} TP {dec['tp']} — {dec['why']}")
            st.session_state.markers.append((c['label'],px,dec["side"]))

# --- UI ---
st.title("🤖 Auto-Trader Bot — Trial v5 (generic intraday)")
st.caption("Swings, BOS, S/R and trades are computed from today's RTH bars only. No overnight levels.")
import datetime
ticker=st.sidebar.text_input("Ticker","QQQ").upper()
day=st.sidebar.date_input("Date", datetime.date.today()-timedelta(days=1))
day_df=fetch_day(ticker,day)
if day_df is not None and not day_df.empty:
    if st.sidebar.button("🚀 Start / Reset"):
        st.session_state.day_df=day_df; st.session_state.step=30
        for k in ["balance","shares","sl","tp","side","log","markers","cooldown"]: 
            st.session_state[k]={"balance":1000.0,"shares":0.0,"sl":None,"tp":None,"side":None,"log":[],"markers":[],"cooldown":0}[k]
        st.session_state.active=True; st.rerun()
else: st.sidebar.error("No 1m data for that date. yfinance keeps 1m only last 7 days.")

if st.session_state.active:
    i=st.session_state.step; df=st.session_state.day_df; vis=df.iloc[:i+1]; price=vis["close"].iloc[-1]
    atr=(df["high"]-df["low"]).iloc[max(0,i-20):i+1].mean()
    swings=get_swings(vis); labeled=label_swings([s for s in swings if s[0]+SWING_K<=i])
    bos=detect_bos(vis,labeled); sup,res=build_sr(labeled,price,atr)
    equity=st.session_state.balance+st.session_state.shares*price
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Price",f"${price:.2f}"); c2.metric("Equity",f"${equity:.2f}",f"${equity-1000:.2f}")
    c3.metric("Position",st.session_state.side or "FLAT"); c4.metric("Last BOS",bos[-1]["dir"] if bos else "None")
    fig=go.Figure([go.Candlestick(x=vis["label"],open=vis["open"],high=vis["high"],low=vis["low"],close=vis["close"],name="price")])
    for s in labeled:
        fig.add_annotation(x=vis["label"].iloc[s["i"]],y=s["price"],text=s["label"],showarrow=False,yshift=-14 if s["type"]=="L" else 14,font=dict(size=10,color="lime" if "H" in s["label"] else "red"))
    for b in bos:
        fig.add_annotation(x=vis["label"].iloc[b["i"]],y=vis["close"].iloc[b["i"]],text="BOS↑" if b["dir"]=="up" else "BOS↓",showarrow=True,arrowhead=2,font=dict(color="yellow",size=11))
        fig.add_shape(type="line",x0=vis["label"].iloc[b["broken_idx"]],x1=vis["label"].iloc[b["i"]],y0=b["broken_price"],y1=b["broken_price"],line=dict(color="yellow",dash="dot",width=1))
    for z in sup: fig.add_hline(y=z["mid"],line=dict(color="lime",width=min(z["touches"],3),dash="dot"),annotation_text=f"S {z['mid']:.2f} ({z['touches']}x)")
    for z in res: fig.add_hline(y=z["mid"],line=dict(color="red",width=min(z["touches"],3),dash="dot"),annotation_text=f"R {z['mid']:.2f} ({z['touches']}x)")
    if st.session_state.side:
        fig.add_hline(y=st.session_state.sl,line=dict(color="orange",dash="dash")); fig.add_hline(y=st.session_state.tp,line=dict(color="cyan",dash="dash"))
    fig.update_layout(template="plotly_dark",height=620,xaxis_rangeslider_visible=False,title=f"{ticker} {day} | {vis['label'].iloc[-1]} | ATR {atr:.2f}")
    fig.update_xaxes(type="category",nticks=12)
    st.plotly_chart(fig,use_container_width=True)
    a1,a2,a3=st.columns(3)
    if a1.button("▶️ +1 Min"): advance(1); st.rerun()
    if a2.button("⏩ +5 Min"): advance(5); st.rerun()
    if a3.button("⏭️ +15 Min"): advance(15); st.rerun()
    if st.session_state.log:
        with st.expander("📝 Bot Decision Log",expanded=True):
            for l in reversed(st.session_state.log): st.text(l)
