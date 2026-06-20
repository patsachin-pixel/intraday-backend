"""
Intraday Signal Backend
------------------------
Fetches Nifty 50 + Nifty Midcap 50 sample stocks from Yahoo Finance,
computes VWAP / RSI / MACD / Volume based intraday signals,
and serves results as JSON over HTTP.

Deploy this on Render (or any host) and point your dashboard at:
  GET /api/signals  -> all stocks with computed signals
  GET /health        -> simple uptime check
"""

import time
import threading
from datetime import datetime, timezone

import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow the phone dashboard (different origin) to call this API

# ---------------- Universe ----------------
# Yahoo Finance uses ".NS" suffix for NSE-listed stocks
LARGE_CAP = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "ITC", "SBIN",
    "LT", "HCLTECH", "AXISBANK", "KOTAKBANK", "MARUTI", "SUNPHARMA", "TITAN",
    "BAJFINANCE", "ULTRACEMCO", "ASIANPAINT", "NESTLEIND", "WIPRO",
]
MID_CAP = [
    "PERSISTENT", "POLYCAB", "COFORGE", "FEDERALBNK", "INDHOTEL", "MPHASIS",
    "ASTRAL", "PIIND", "AUBANK", "VOLTAS", "BHARATFORG", "TATACOMM",
    "SUPREMEIND", "CONCOR", "IPCALAB", "GODREJPROP", "LUPIN", "BALKRISIND",
]
UNIVERSE = [(s, "Large") for s in LARGE_CAP] + [(s, "Mid") for s in MID_CAP]

POLL_SECONDS = 30  # Yahoo doesn't truly support 5-sec polling; 30s is realistic & won't get rate-limited

# Shared in-memory cache, updated by the background thread
_cache = {"updated_at": None, "stocks": []}
_cache_lock = threading.Lock()


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty and not np.isnan(rsi.iloc[-1]) else 50.0


def compute_macd_hist(close: pd.Series) -> float:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    return float(hist.iloc[-1]) if not hist.empty else 0.0


def compute_vwap(df: pd.DataFrame) -> float:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return float(vwap.iloc[-1]) if not vwap.empty else float(df["Close"].iloc[-1])


def compute_signal(price, vwap, rsi, macd_hist, volume, avg_volume):
    vwap_signal = 1 if price > vwap else -1
    rsi_signal = 1 if rsi < 35 else (-1 if rsi > 65 else 0)
    macd_signal = 1 if macd_hist > 0 else -1
    vol_ratio = volume / avg_volume if avg_volume else 1.0
    vol_signal = (1 if vwap_signal > 0 else -1) if vol_ratio > 1.3 else 0

    signals = [vwap_signal, rsi_signal, macd_signal, vol_signal]
    score = sum(signals)
    agreeing = [s for s in signals if s != 0]
    bull = sum(1 for s in agreeing if s > 0)
    bear = sum(1 for s in agreeing if s < 0)
    confidence = (max(bull, bear) / 4) if agreeing else 0.0

    action = "WATCH"
    if score >= 2 and bull >= 3:
        action = "BUY"
    elif score <= -2 and bear >= 3:
        action = "SELL"

    target_pct = (0.4 + confidence * 0.8) if action == "BUY" else (-(0.4 + confidence * 0.8) if action == "SELL" else 0)
    stop_pct = -0.5 if action == "BUY" else (0.5 if action == "SELL" else 0)

    return {
        "action": action,
        "confidence": round(confidence, 2),
        "target": round(price * (1 + target_pct / 100), 2),
        "stop": round(price * (1 + stop_pct / 100), 2),
        "vol_ratio": round(vol_ratio, 2),
    }


def fetch_one(symbol: str, cap: str):
    ticker = yf.Ticker(f"{symbol}.NS")
    df = ticker.history(period="2d", interval="5m")
    if df.empty or len(df) < 20:
        return None

    close = df["Close"]
    price = float(close.iloc[-1])
    prev_close_row = ticker.fast_info.get("previousClose", None) if hasattr(ticker, "fast_info") else None
    prev_close = float(prev_close_row) if prev_close_row else float(close.iloc[0])

    today = df[df.index.date == df.index[-1].date()]
    avg_volume = float(df["Volume"].mean()) or 1.0
    today_volume = float(today["Volume"].sum()) if not today.empty else float(df["Volume"].iloc[-1])

    rsi = compute_rsi(close)
    macd_hist = compute_macd_hist(close)
    vwap = compute_vwap(today if not today.empty else df)

    sig = compute_signal(price, vwap, rsi, macd_hist, today_volume, avg_volume)
    change_pct = ((price - prev_close) / prev_close) * 100 if prev_close else 0.0

    spark = [round(float(v), 2) for v in close.tail(20).tolist()]

    return {
        "sym": symbol,
        "cap": cap,
        "price": round(price, 2),
        "prevClose": round(prev_close, 2),
        "changePct": round(change_pct, 2),
        "rsi": round(rsi, 1),
        "vwap": round(vwap, 2),
        "macdHist": round(macd_hist, 3),
        "volume": int(today_volume),
        "avgVolume": int(avg_volume),
        "spark": spark,
        "signal": sig,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def refresh_loop():
    while True:
        results = []
        for symbol, cap in UNIVERSE:
            try:
                row = fetch_one(symbol, cap)
                if row:
                    results.append(row)
            except Exception as e:
                print(f"[warn] failed to fetch {symbol}: {e}")
            time.sleep(0.3)  # be polite to Yahoo's endpoint between symbols

        with _cache_lock:
            _cache["stocks"] = results
            _cache["updated_at"] = datetime.now(timezone.utc).isoformat()

        print(f"[info] refreshed {len(results)} stocks at {_cache['updated_at']}")
        time.sleep(POLL_SECONDS)


@app.route("/api/signals")
def get_signals():
    with _cache_lock:
        return jsonify({
            "updated_at": _cache["updated_at"],
            "count": len(_cache["stocks"]),
            "stocks": _cache["stocks"],
        })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "last_refresh": _cache["updated_at"]})


@app.route("/")
def index():
    return jsonify({
        "service": "Intraday Signal Backend",
        "endpoints": ["/api/signals", "/health"],
        "note": "Data sourced from Yahoo Finance, ~30s refresh. Not real-time tick data.",
    })


# Start background refresh thread once, on import
_thread = threading.Thread(target=refresh_loop, daemon=True)
_thread.start()

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
