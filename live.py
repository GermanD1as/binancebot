"""
live.py — Binance Testnet trading bot.
Ejecuta Grid o RSI en BNB/USDT 15m con filtro de tendencia 1h.

Uso:
  export BINANCE_TESTNET_API_KEY=xxx
  export BINANCE_TESTNET_API_SECRET=yyy
  python live.py                    # Grid:BNB default
  python live.py --strategy rsi     # RSI:BNB
  python live.py --dry-run          # sin ejecutar órdenes
"""

import argparse
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
import requests

from backtest import fetch_ohlcv, add_indicators, add_mtf_trend

def _git_out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace").stdout.strip()
    except Exception:
        return "?"
GIT_REV = _git_out(["git", "rev-parse", "--short", "HEAD"])
GIT_DATE = _git_out(["git", "log", "-1", "--format=%ci"])

def _notify(msg: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": msg, "parse_mode": "HTML"},
                          timeout=5)
        except Exception:
            pass

CACHE_DIR = "cache"
CONFIG = {
    "pair": "BNB/USDT",
    "interval_base": "15m",
    "interval_trend": "1h",
    "days_base": 2,
    "days_trend": 15,
    "grid_pct": 0.01,
    "n_levels": 3,
    "tp_mult": 1.5,
    "stop_loss_pct": 0.02,
    "rsi_os": 30,
    "rsi_ob": 70,
    "commission": 0.001,
    "slippage": 0.0005,
    "slippage_atr_factor": 0.1,
}


class LiveBot:
    def __init__(self, strategy: str, dry_run: bool = False):
        self.strategy = strategy
        self.dry_run = dry_run
        self.pair = CONFIG["pair"]
        self.state_path = f"{CACHE_DIR}/live_{strategy}_{self.pair.replace('/', '_')}.pkl"

        if not dry_run:
            key = os.environ.get("BINANCE_TESTNET_API_KEY")
            secret = os.environ.get("BINANCE_TESTNET_API_SECRET")
            if not key or not secret:
                print("ERROR: Set BINANCE_TESTNET_API_KEY y BINANCE_TESTNET_API_SECRET")
                sys.exit(1)
            self.exchange = ccxt.binance({
                "apiKey": key, "secret": secret,
                "enableRateLimit": True, "options": {"defaultType": "spot"},
            })
            proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
            if proxy:
                self.exchange.proxies = {"https": proxy, "http": proxy}
            self.exchange.set_sandbox_mode(True)

        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.state_path):
            with open(self.state_path, "rb") as f:
                return pickle.load(f)
        return {"cash": 1000.0, "positions": {}, "pnl_history": []}

    def _save_state(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.state_path, "wb") as f:
            pickle.dump(self.state, f)

    def _price(self, px: float, side: str, atr: float | None = None) -> float:
        slip = max(CONFIG["slippage"], (atr / px * CONFIG["slippage_atr_factor"]) if atr else 0)
        return px * (1 + slip) if side == "buy" else px * (1 - slip)

    def _order(self, side: str, size: float, price: float):
        if self.dry_run:
            print(f"  [DRY] {side.upper()} {size:.4f} @ ${price:.2f}")
            return {"price": price, "filled": size}
        try:
            return self.exchange.create_market_order(self.pair, side, size)
        except ccxt.InsufficientFunds as e:
            print(f"  ⚠ Fondos insuficientes: {e}")
            return None
        except ccxt.NetworkError as e:
            print(f"  ⚠ Error de red al ordenar: {e}")
            return None

    def run(self):
        df = add_indicators(fetch_ohlcv(self.pair, CONFIG["interval_base"], CONFIG["days_base"], use_cache=False))
        df = add_mtf_trend(df, self.pair, CONFIG["days_trend"])
        row = df.iloc[-1]
        price = float(row["close"])
        trend = int(row["trend_1h"])
        rsi = float(row["rsi"])
        ema20 = float(row["ema20"])
        atr = float(row["atr"]) if "atr" in row else 0.0
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        n_trades = len(self.state.get("pnl_history", []))

        print(f"[{ts}] v{GIT_REV} | {self.strategy} | cash=${self.state['cash']:.2f} | pos={len(self.state['positions'])}")
        print(f"[{ts}] {self.pair} ${price:.2f}  trend={trend}  rsi={rsi:.0f}  ema20={ema20:.2f}  sep={abs(price/ema20-1)*100:.2f}%  atr={atr:.4f}")

        if self.strategy == "grid":
            no_signal = self._grid(price, ema20, trend, row, atr)
        elif self.strategy == "rsi":
            no_signal = self._rsi(price, trend, rsi, atr)

        if no_signal and n_trades == len(self.state.get("pnl_history", [])):
            print(f"  Sin señal — esperando oportunidad")
        self._report()
        self._save_state()

    def run_safe(self):
        try:
            self.run()
        except Exception as e:
            import traceback
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            print(f"[{ts}] ERROR: {e}")
            traceback.print_exc()
            _notify(f"⚠️ {self.strategy.upper()} ERROR: {e}")
            self._save_state()

    # ── GRID ──
    def _grid(self, price: float, ema20: float, trend: int, row, atr: float = 0):
        pos = self.state["positions"]
        sp = CONFIG["grid_pct"]

        for key, p in list(pos.items()):
            if p["side"] == "long" and (price >= p["tp"] or price <= p["sl"]):
                self._close(key, price, "TP" if price >= p["tp"] else "SL", atr)
            elif p["side"] == "short" and (price <= p["tp"] or price >= p["sl"]):
                self._close(key, price, "TP" if price <= p["tp"] else "SL", atr)

        entered = False
        if trend != -1:
            for j in range(CONFIG["n_levels"]):
                k = f"B{j}"
                if k not in pos and price <= ema20 * (1 - sp * (j + 1)):
                    self._open(k, "long", price, atr)
                    entered = True
        if trend != 1:
            for j in range(CONFIG["n_levels"]):
                k = f"S{j}"
                if k not in pos and price >= ema20 * (1 + sp * (j + 1)):
                    self._open(k, "short", price, atr)
                    entered = True
        return not entered and not pos

    # ── RSI ──
    def _rsi(self, price: float, trend: int, rsi: float, atr: float = 0):
        pos = self.state["positions"]
        entered = False

        for key, p in list(pos.items()):
            sl = p["entry"] * (1 - CONFIG["stop_loss_pct"]) if p["side"] == "long" else p["entry"] * (1 + CONFIG["stop_loss_pct"])
            exit_rsi = (p["side"] == "long" and rsi > CONFIG["rsi_ob"]) or (p["side"] == "short" and rsi < CONFIG["rsi_os"])
            exit_sl = (p["side"] == "long" and price <= sl) or (p["side"] == "short" and price >= sl)
            if exit_sl:
                self._close(key, price, "SL", atr)
            elif exit_rsi:
                self._close(key, price, "RSI", atr)

        if not pos:
            if trend != -1 and rsi < CONFIG["rsi_os"]:
                self._open("LONG", "long", price, atr)
                entered = True
            elif trend != 1 and rsi > CONFIG["rsi_ob"]:
                self._open("SHORT", "short", price, atr)
                entered = True
        return not entered and not pos

    # ── Order helpers ──
    def _open(self, key: str, side: str, price: float, atr: float = 0):
        cash = self.state["cash"]
        if cash < 10:
            return
        entry = self._price(price, "buy" if side == "long" else "sell", atr)
        if side == "long":
            size = (cash * (1 - CONFIG["commission"])) / entry
            new_cash = 0.0
        else:
            size = cash / entry
            new_cash = cash + size * entry * (1 - CONFIG["commission"])
        result = self._order("buy" if side == "long" else "sell", size, entry)
        if result is None:
            return
        self.state["cash"] = new_cash
        tp = entry * (1 + CONFIG["tp_mult"] * CONFIG["grid_pct"]) if side == "long" else entry * (1 - CONFIG["tp_mult"] * CONFIG["grid_pct"])
        sl = entry * (1 - CONFIG["stop_loss_pct"]) if side == "long" else entry * (1 + CONFIG["stop_loss_pct"])
        self.state["positions"][key] = {"entry": entry, "size": size, "side": side, "tp": tp, "sl": sl}
        print(f"  → OPEN {key} {side.upper()} @ ${entry:.2f}  sz={size:.4f}  cash=${self.state['cash']:.2f}")
        _notify(f"🟢 {self.strategy.upper()} OPEN {key} {side} BNB @ ${entry:.2f}")

    def _close(self, key: str, price: float, reason: str, atr: float = 0):
        p = self.state["positions"].get(key)
        if not p:
            return
        exit_px = self._price(price, "sell" if p["side"] == "long" else "buy", atr)
        result = self._order("sell" if p["side"] == "long" else "buy", p["size"], exit_px)
        if result is None:
            return
        del self.state["positions"][key]
        if p["side"] == "long":
            proceeds = p["size"] * exit_px * (1 - CONFIG["commission"])
            buy_cost = p["size"] * p["entry"] * (1 + CONFIG["commission"])
            pnl = proceeds - buy_cost
            self.state["cash"] = proceeds
        else:
            buy_cost = p["size"] * exit_px * (1 + CONFIG["commission"])
            proceeds_in = p["size"] * p["entry"] * (1 - CONFIG["commission"])
            pnl = proceeds_in - buy_cost
            self.state["cash"] = self.state["cash"] - buy_cost
        self.state.setdefault("pnl_history", []).append(pnl)
        print(f"  ← CLOSE {key} {p['side'].upper()} @ ${exit_px:.2f}  PnL=${pnl:.2f}  cash=${self.state['cash']:.2f}  ({reason})")
        emoji = "🔴" if pnl < 0 else "🟢"
        _notify(f"{emoji} {self.strategy.upper()} CLOSE {key} {p['side']} BNB @ ${exit_px:.2f} | PnL=${pnl:.2f} ({reason})")

    def _report(self):
        val = self.state["cash"]
        for p in self.state["positions"].values():
            val += p["size"] * p["entry"]
        n_pos = len(self.state["positions"])
        hist = self.state.get("pnl_history", [])
        total_pnl = sum(hist)
        n_trades = len(hist)
        print(f"  ── Portfolio: ${val:.2f}  |  {n_pos} pos  |  Trades: {n_trades}  |  PnL: ${total_pnl:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BNB Testnet Bot")
    parser.add_argument("--strategy", default="grid", choices=["grid", "rsi"])
    parser.add_argument("--dry-run", action="store_true", help="No ejecuta órdenes reales")
    args = parser.parse_args()
    LiveBot(args.strategy, args.dry_run).run_safe()
