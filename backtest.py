from __future__ import annotations
"""
╔══════════════════════════════════════════════════════════════╗
║           CRYPTO BACKTEST FRAMEWORK — v1.1                  ║
║  Descarga datos reales de Binance y compara estrategias      ║
╚══════════════════════════════════════════════════════════════╝

Instalación:
    pip3 install ccxt pandas ta tabulate colorama

Uso:
    python backtest.py                        # configuración por defecto
    python backtest.py --pair ETHUSDT         # otro par
    python backtest.py --interval 4h          # velas de 4 horas
    python backtest.py --days 365             # último año
    python backtest.py --capital 500          # capital inicial
    python backtest.py --optimize             # optimiza parámetros del ganador
"""

import argparse
import logging
import os
import pickle
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

try:
    import ccxt
except ImportError:
    print("❌  Falta ccxt. Instalá con:  pip install ccxt")
    sys.exit(1)

try:
    import ta as ta_lib
except ImportError:
    print("❌  Falta ta. Instalá con:  pip3 install ta")
    sys.exit(1)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    GREEN  = Fore.GREEN
    RED    = Fore.RED
    YELLOW = Fore.YELLOW
    CYAN   = Fore.CYAN
    BOLD   = Style.BRIGHT
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""


# ─────────────────────────────────────────────────────────────
# 1. DESCARGA DE DATOS
# ─────────────────────────────────────────────────────────────

CACHE_DIR = "cache"

def _cache_path(pair: str, interval: str, days: int) -> str:
    safe = pair.replace("/", "_")
    return f"{CACHE_DIR}/{safe}_{interval}_{days}d.pkl"

def _cache_valid(path: str) -> bool:
    today = datetime.now(timezone.utc).date()
    return os.path.exists(path) and datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).date() == today

def fetch_ohlcv(pair: str, interval: str, days: int, use_cache: bool = True) -> pd.DataFrame:
    """Descarga datos históricos de Binance vía ccxt, con caché."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = _cache_path(pair, interval, days)

    if use_cache and _cache_valid(cache_path):
        with open(cache_path, "rb") as f:
            df = pickle.load(f)
        print(f"{CYAN}📦  Cargando {pair} [{interval}] desde caché ({len(df)} velas){RESET}")
        return df

    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    print(f"{CYAN}⬇  Descargando {pair} [{interval}] — últimos {days} días...{RESET}")

    all_ohlcv = []
    while True:
        for attempt in range(3):
            try:
                batch = exchange.fetch_ohlcv(pair, interval, since=since_ms, limit=1000)
                break
            except (ccxt.NetworkError, ccxt.RateLimitError) as e:
                if attempt == 2:
                    print(f"{RED} Error tras 3 intentos: {e}{RESET}")
                    raise
                time.sleep(2 ** attempt)
        if not batch:
            break
        all_ohlcv.extend(batch)
        since_ms = batch[-1][0] + 1
        if len(batch) < 1000:
            break

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)

    print(f"{GREEN}✔  {len(df)} velas descargadas  ({df.index[0].date()} → {df.index[-1].date()}){RESET}\n")
    return df


# ─────────────────────────────────────────────────────────────
# 2. INDICADORES TÉCNICOS
# ─────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega todos los indicadores necesarios al DataFrame."""
    df = df.copy()
    c = df["close"]

    # RSI (14 y 5 para scalping)
    df["rsi"]  = ta_lib.momentum.RSIIndicator(c, window=14).rsi()
    df["rsi5"] = ta_lib.momentum.RSIIndicator(c, window=5).rsi()

    # EMAs (largas y cortas)
    for w in [5, 8, 12, 13, 20, 26, 50]:
        df[f"ema{w}"] = ta_lib.trend.EMAIndicator(c, window=w).ema_indicator()

    # Bollinger Bands
    bb = ta_lib.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()

    # ATR (14 y 7 para scalping)
    df["atr"]  = ta_lib.volatility.AverageTrueRange(df["high"], df["low"], c, window=14).average_true_range()
    df["atr7"] = ta_lib.volatility.AverageTrueRange(df["high"], df["low"], c, window=7).average_true_range()

    # VWAP rolling 20 períodos (en lugar de acumulado, para señales significativas)
    tp = (df["high"] + df["low"] + c) / 3
    df["vwap"] = (tp * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    vwap_std = c.rolling(20).std()
    df["vwap_upper"] = df["vwap"] + 2 * vwap_std
    df["vwap_lower"] = df["vwap"] - 2 * vwap_std

    # Volumen MA
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    df.dropna(inplace=True)
    return df


def add_mtf_trend(df: pd.DataFrame, pair: str, days: int) -> pd.DataFrame:
    """Agrega filtro direccional desde 1h (EMA50/EMA200) al df base.
    
    trend_1h = 1 (bull) si EMA50_1h > EMA200_1h, -1 (bear) en caso contrario.
    trend_strength = pendiente diaria normalizada de EMA50_1h.
    """
    df_h1 = fetch_ohlcv(pair, "1h", days + 30)
    c_h1 = df_h1["close"]
    df_h1["ema50_1h"] = ta_lib.trend.EMAIndicator(c_h1, window=50).ema_indicator()
    df_h1["ema200_1h"] = ta_lib.trend.EMAIndicator(c_h1, window=200).ema_indicator()
    df_h1["trend_1h"] = np.where(df_h1["ema50_1h"] > df_h1["ema200_1h"], 1, -1)
    # Pendiente de EMA50: % de cambio en 24 horas
    df_h1["ema50_slope"] = df_h1["ema50_1h"].diff(24) / df_h1["ema50_1h"].shift(24)
    cols = ["trend_1h", "ema50_slope"]
    merged = df_h1[cols].reindex(df.index, method="ffill")
    df = df.join(merged, how="left")
    df["trend_1h"] = df["trend_1h"].fillna(0).astype(int)
    df["trend_strength"] = df["ema50_slope"].fillna(0)
    df.drop(columns=["ema50_slope"], inplace=True)
    return df


def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Clasifica régimen de mercado por barra (bull/bear/sideways/high_vol).
    
    Requiere: trend_1h, trend_strength, atr.
    """
    atr = df["atr"]
    atr_ma = atr.rolling(504).mean()
    atr_std = atr.rolling(504).std()
    df["atr_z"] = (atr - atr_ma) / (atr_std + 1e-10)

    conditions = [
        (df["atr_z"] > 2.0),
        (df["trend_1h"] == 1) & (df["trend_strength"] > 0.001),
        (df["trend_1h"] == 1),
        (df["trend_1h"] == -1) & (df["trend_strength"] < -0.001),
        (df["trend_1h"] == -1),
    ]
    choices = ["high_vol", "bull_strong", "bull_weak", "bear_strong", "bear_weak"]
    df["regime"] = np.select(conditions, choices, default="sideways")
    return df


# ─────────────────────────────────────────────────────────────
# 3. MOTOR DE BACKTESTING
# ─────────────────────────────────────────────────────────────

def annualization_factor(interval: str) -> float:
    """Devuelve sqrt(periodos_por_año) para anualizar Sharpe según el intervalo."""
    minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
               "1h": 60, "2h": 120, "4h": 240, "6h": 360,
               "8h": 480, "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080}
    periods = (365 * 24 * 60) / minutes.get(interval, 1440)
    return np.sqrt(periods)

class BacktestEngine:
    def __init__(self, df: pd.DataFrame, capital: float = 500.0, commission: float = 0.001,
                 slippage: float = 0.0005, stop_loss_pct: float = 0.0,
                 allow_short: bool = False, interval_factor: float = np.sqrt(252),
                 min_notional: float = 10.0):
        self.df = df
        self.initial_capital = capital
        self.commission = commission
        self.slippage = slippage
        self.stop_loss_pct = stop_loss_pct
        self.allow_short = allow_short
        self.interval_factor = interval_factor
        self.min_notional = min_notional

    def run(self, signals: pd.Series) -> dict:
        cash     = self.initial_capital
        position = 0.0
        entry_px = 0.0
        equity   = []
        trades   = []
        bars_in_market = 0

        for i, (ts, row) in enumerate(self.df.iterrows()):
            price  = row["close"]
            signal = signals.iloc[i]
            trend = row.get("trend_1h", 0)

            # ── Stop-loss ──
            if position > 0 and self.stop_loss_pct > 0:
                stop = entry_px * (1 - self.stop_loss_pct)
                if price <= stop:
                    proceeds = position * stop * (1 - self.commission)
                    pnl = proceeds - (position * entry_px)
                    trades.append({
                        "date": ts, "entry": entry_px, "exit": stop,
                        "pnl": pnl, "pnl_pct": pnl / (position * entry_px) * 100,
                    })
                    cash, position, entry_px = proceeds, 0.0, 0.0
            elif position < 0 and self.stop_loss_pct > 0:
                stop = abs(entry_px) * (1 + self.stop_loss_pct)
                if price >= stop:
                    units = abs(position)
                    buy_cost = units * stop * (1 + self.commission)
                    sell_proceeds = units * abs(entry_px) * (1 - self.commission)
                    pnl = sell_proceeds - buy_cost
                    trades.append({
                        "date": ts, "entry": entry_px, "exit": stop,
                        "pnl": pnl, "pnl_pct": pnl / (units * abs(entry_px)) * 100,
                    })
                    cash = cash - buy_cost
                    position, entry_px = 0.0, 0.0

            # ── Señales de trading (if/elif para evitar abrir y cerrar en la misma barra) ──
            if signal == -1 and position > 0:
                sell_px = price * (1 - self.slippage)
                proceeds = position * sell_px * (1 - self.commission)
                pnl = proceeds - (position * entry_px)
                trades.append({
                    "date": ts, "entry": entry_px, "exit": sell_px,
                    "pnl": pnl, "pnl_pct": pnl / (position * entry_px) * 100,
                })
                cash = proceeds
                position, entry_px = 0.0, 0.0

            elif signal == -1 and cash > 0 and position == 0 and self.allow_short and cash >= self.min_notional and trend != 1:
                sell_px = price * (1 - self.slippage)
                units = cash / sell_px
                proceeds = units * sell_px * (1 - self.commission)
                position = -units
                cash = cash + proceeds
                entry_px = sell_px

            elif signal == 1 and position < 0:
                units = abs(position)
                buy_px = price * (1 + self.slippage)
                buy_cost = units * buy_px * (1 + self.commission)
                sell_proceeds = units * abs(entry_px) * (1 - self.commission)
                pnl = sell_proceeds - buy_cost
                trades.append({
                    "date": ts, "entry": entry_px, "exit": buy_px,
                    "pnl": pnl, "pnl_pct": pnl / (units * abs(entry_px)) * 100,
                })
                cash = cash - buy_cost
                position, entry_px = 0.0, 0.0

            elif signal == 1 and cash > 0 and position == 0 and trend != -1:
                buy_px = price * (1 + self.slippage)
                if cash * (1 - self.commission) >= self.min_notional:
                    units = (cash * (1 - self.commission)) / buy_px
                    position = units
                    cash = 0.0
                    entry_px = buy_px

            if position != 0:
                bars_in_market += 1
            equity.append(cash + position * price)

        # ── Cerrar posición al final ──
        if position > 0:
            lp = self.df["close"].iloc[-1]
            sp = lp * (1 - self.slippage)
            proceeds = position * sp * (1 - self.commission)
            equity[-1] = proceeds
        elif position < 0:
            lp = self.df["close"].iloc[-1]
            bp = lp * (1 + self.slippage)
            units = abs(position)
            buy_cost = units * bp * (1 + self.commission)
            equity[-1] = cash - buy_cost

        return self._metrics(equity, trades, bars_in_market)

    def _metrics(self, equity: list, trades: list, bars_in_market: int = 0) -> dict:
        eq  = np.array(equity, dtype=float)
        ret = np.diff(eq) / (eq[:-1] + 1e-10)

        total_return = (eq[-1] - self.initial_capital) / self.initial_capital * 100
        final_capital = eq[-1]

        # Sharpe (anualizado según intervalo)
        sharpe = (ret.mean() / (ret.std() + 1e-10)) * self.interval_factor if len(ret) > 1 else 0.0

        # Max Drawdown & Ulcer Index
        peak    = np.maximum.accumulate(eq)
        dd      = (eq - peak) / peak * 100
        max_dd  = float(dd.min())
        dd_sq   = np.nan_to_num((dd / 100) ** 2)
        ulcer_index = float(np.sqrt(np.mean(dd_sq)))

        # Calmar Ratio
        calmar = abs(total_return / max_dd) if max_dd != 0 else 0.0

        # Métricas de trades
        n_trades = len(trades)
        winners  = [t for t in trades if t["pnl"] > 0]
        win_rate = len(winners) / n_trades * 100 if n_trades else 0.0

        gross_profit = sum(t["pnl"] for t in winners) or 0.0
        gross_loss   = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0)) or 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # SQN
        trade_pnls = np.array([t["pnl"] for t in trades]) if trades else np.array([0.0])
        sqn = (trade_pnls.mean() / (trade_pnls.std() + 1e-10)) * np.sqrt(n_trades) if n_trades > 1 else 0.0

        # Expectancy
        expectancy = float(trade_pnls.mean()) if n_trades else 0.0

        # Time in Market
        time_in_market_pct = (bars_in_market / len(eq)) * 100 if len(eq) > 1 else 0.0

        return {
            "equity":          eq,
            "drawdowns":       dd,
            "trades":          trades,
            "total_return":    total_return,
            "sharpe":          sharpe,
            "max_drawdown":    max_dd,
            "ulcer_index":     ulcer_index,
            "calmar":          calmar,
            "win_rate":        win_rate,
            "profit_factor":   profit_factor,
            "sqn":             sqn,
            "expectancy":      expectancy,
            "n_trades":        n_trades,
            "final_capital":   final_capital,
            "time_in_market":  time_in_market_pct,
        }


# ─────────────────────────────────────────────────────────────
# 3b. GRID ENGINE (multi-nivel)
# ─────────────────────────────────────────────────────────────

class GridBacktestEngine:
    """Grid market-making simétrico: compra bajo EMA20, vende (short) sobre EMA20.
    
    Si atr_mult > 0, el spacing se adapta dinámicamente como atr_mult × ATR / precio.
    Si atr_mult = 0 (default), usa grid_pct fijo.
    """

    def __init__(self, df: pd.DataFrame, capital: float = 1000.0, commission: float = 0.001,
                 slippage: float = 0.0005, grid_pct: float = 0.01, n_levels: int = 10,
                 interval_factor: float = np.sqrt(252), min_notional: float = 10.0,
                 stop_loss_pct: float = 0.0, allow_short: bool = False,
                 tp_mult: float = 1.5, atr_mult: float = 0.0):
        self.df = df
        self.initial_capital = capital
        self.commission = commission
        self.slippage = slippage
        self.grid_pct = grid_pct
        self.n_levels = n_levels
        self.interval_factor = interval_factor
        self.min_notional = min_notional
        self.stop_loss_pct = stop_loss_pct if stop_loss_pct > 0 else grid_pct
        self.allow_short = allow_short
        self.tp_mult = tp_mult
        self.atr_mult = atr_mult
        if "atr" in df.columns:
            self.atr = df["atr"].values
        else:
            self.atr = None

    def _spacing(self, i: int, ref_val: float) -> float:
        """Retorna el spacing dinámico (ATR-based o fijo) para la barra i."""
        if self.atr_mult > 0 and self.atr is not None:
            a = self.atr[i]
            if not np.isnan(a) and a > 0 and ref_val > 0:
                return float(np.clip(self.atr_mult * a / ref_val, 0.001, 0.05))
        return self.grid_pct

    def run(self) -> dict:
        positions = {}
        trades = []
        equity = []
        realized_pnl = 0.0

        n_buy = self.n_levels
        n_sell = self.n_levels if self.allow_short else 0
        n_total = n_buy + n_sell
        if n_total == 0:
            return self._metrics(np.array([self.initial_capital], dtype=float), [])

        per_level = (self.initial_capital * 0.9) / n_total
        close = self.df["close"].values
        if "trend_1h" in self.df.columns:
            trend_arr = self.df["trend_1h"].values
        else:
            trend_arr = None
        if "ema20" in self.df.columns:
            ref = np.nan_to_num(self.df["ema20"].values, nan=close[0])
        else:
            ref = pd.Series(close).rolling(20).mean().fillna(method="bfill").values

        for i in range(1, len(self.df)):
            price = close[i]
            prev = close[i - 1]
            r = ref[i]
            if np.isnan(r) or r == 0:
                equity.append(self.initial_capital + realized_pnl)
                continue

            sp = self._spacing(i, r)
            tp_rate = sp * self.tp_mult
            sl_rate_long = sp * (1 if self.stop_loss_pct == 0 else self.stop_loss_pct / self.grid_pct)
            if self.stop_loss_pct > 0:
                sl_rate = self.stop_loss_pct
            else:
                sl_rate = sp

            buy_levels = [r * (1 - sp * (j + 1)) for j in range(n_buy)]
            trend = trend_arr[i] if trend_arr is not None else 0
            for j, lv in enumerate(buy_levels):
                key = f"B{j}"
                if key not in positions and prev > lv >= price and trend != -1:
                    size = (per_level * (1 - self.commission)) / (price * (1 + self.slippage))
                    positions[key] = {"entry": price, "size": size, "side": "long",
                                      "tp": price * (1 + tp_rate),
                                      "sl": price * (1 - sl_rate)}

            sell_levels = [r * (1 + sp * (j + 1)) for j in range(n_sell)]
            for j, lv in enumerate(sell_levels):
                key = f"S{j}"
                if key not in positions and prev < lv <= price and trend != 1:
                    size = (per_level * (1 - self.commission)) / (price * (1 - self.slippage))
                    positions[key] = {"entry": price, "size": size, "side": "short",
                                      "tp": price * (1 - tp_rate),
                                      "sl": price * (1 + sl_rate)}

            for key, pos in list(positions.items()):
                if pos["side"] == "long":
                    if price >= pos["tp"] or price <= pos["sl"]:
                        exit_px = price * (1 - self.slippage)
                        proceeds = pos["size"] * exit_px * (1 - self.commission)
                        pnl = proceeds - (pos["size"] * pos["entry"])
                        reason = "TP" if price >= pos["tp"] else "SL"
                        trades.append(self._trade(self.df.index[i], pos, exit_px, pnl, key, reason))
                        realized_pnl += pnl
                        del positions[key]
                else:
                    if price <= pos["tp"] or price >= pos["sl"]:
                        buy_px = price * (1 + self.slippage)
                        buy_cost = pos["size"] * buy_px * (1 + self.commission)
                        pnl = (pos["size"] * pos["entry"]) - buy_cost
                        reason = "TP" if price <= pos["tp"] else "SL"
                        trades.append(self._trade(self.df.index[i], pos, buy_px, pnl, key, reason))
                        realized_pnl += pnl
                        del positions[key]

            unrealized = sum(p["size"] * (price - p["entry"]) if p["side"] == "long"
                             else p["size"] * (p["entry"] - price)
                             for p in positions.values())
            equity.append(self.initial_capital + realized_pnl + unrealized)

        # Cerrar posiciones al final
        final_pnl = realized_pnl
        for pos in positions.values():
            lp = close[-1]
            if pos["side"] == "long":
                sp = lp * (1 - self.slippage)
                final_pnl += pos["size"] * sp * (1 - self.commission) - pos["size"] * pos["entry"]
            else:
                bp = lp * (1 + self.slippage)
                final_pnl -= pos["size"] * bp * (1 + self.commission) - pos["size"] * pos["entry"]
        equity[-1] = self.initial_capital + final_pnl

        return self._metrics(equity, trades)

    def _metrics(self, equity: list, trades: list) -> dict:
        eq = np.array(equity, dtype=float)
        ret = np.diff(eq) / (eq[:-1] + 1e-10)
        total_return = (eq[-1] - self.initial_capital) / self.initial_capital * 100
        final_capital = eq[-1]
        sharpe = (ret.mean() / (ret.std() + 1e-10)) * self.interval_factor if len(ret) > 1 else 0.0
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        max_dd = float(dd.min())
        n_trades = len(trades)
        winners = [t for t in trades if t["pnl"] > 0]
        win_rate = len(winners) / n_trades * 100 if n_trades else 0.0
        gp = sum(t["pnl"] for t in winners) or 0.0
        gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0)) or 0.0
        pf = gp / gl if gl > 0 else float("inf")
        tp = np.array([t["pnl"] for t in trades]) if trades else np.array([0.0])
        sqn = (tp.mean() / (tp.std() + 1e-10)) * np.sqrt(n_trades) if n_trades > 1 else 0.0
        dd_sq = np.nan_to_num((dd / 100) ** 2)
        return {
            "equity": eq, "drawdowns": dd, "trades": trades,
            "total_return": total_return, "sharpe": sharpe, "max_drawdown": max_dd,
            "ulcer_index": float(np.sqrt(np.mean(dd_sq))),
            "calmar": abs(total_return / max_dd) if max_dd != 0 else 0.0,
            "win_rate": win_rate, "profit_factor": pf, "sqn": sqn,
            "expectancy": float(tp.mean()), "n_trades": n_trades,
            "final_capital": final_capital, "time_in_market": 0.0,
        }

    @staticmethod
    def _trade(ts, pos, exit_px, pnl, level, reason):
        return {
            "date": ts, "entry": pos["entry"], "exit": exit_px,
            "pnl": pnl, "pnl_pct": pnl / (pos["size"] * pos["entry"]) * 100,
            "level": level, "reason": reason, "side": pos["side"],
        }


# ─────────────────────────────────────────────────────────────
# 3c. REGIME ENGINE (adaptativo)
# ─────────────────────────────────────────────────────────────

class RegimeEngine:
    """Un solo engine que adapta entrada/salida al régimen de mercado.
    
    Regímenes:
      bull_strong  → grid long (ATR-spacing, TP/SL, múltiples niveles)
      bull_weak    → RSI mean reversion long
      bear_strong  → grid short
      bear_weak    → RSI mean reversion short
      sideways     → RSI ambos lados
      high_vol     → sin entradas (spacing demasiado ancho)
    """

    def __init__(self, df: pd.DataFrame, capital: float = 1000.0, commission: float = 0.001,
                 slippage: float = 0.0005, interval_factor: float = np.sqrt(252 * 96),
                 stop_loss_pct: float = 0.02, min_notional: float = 10.0,
                 tp_mult: float = 1.5, atr_mult: float = 2.0,
                 rsi_oversold: int = 30, rsi_overbought: int = 70,
                 n_levels: int = 3):
        self.df = df
        self.initial_capital = capital
        self.commission = commission
        self.slippage = slippage
        self.interval_factor = interval_factor
        self.stop_loss_pct = stop_loss_pct if stop_loss_pct > 0 else 0.02
        self.min_notional = min_notional
        self.tp_mult = tp_mult
        self.atr_mult = atr_mult
        self.rsi_os = rsi_oversold
        self.rsi_ob = rsi_overbought
        self.n_levels = n_levels

    def _spacing(self, i: int, ref_val: float, atr_arr) -> float:
        if self.atr_mult > 0 and atr_arr is not None and not np.isnan(atr_arr[i]) and ref_val > 0:
            return float(np.clip(self.atr_mult * atr_arr[i] / ref_val, 0.001, 0.05))
        return 0.01

    def run(self) -> dict:
        df = self.df
        close = df["close"].values
        rsi = df["rsi"].values if "rsi" in df.columns else None
        atr = df["atr"].values if "atr" in df.columns else None
        regimes = df["regime"].values if "regime" in df.columns else None

        if "ema20" in df.columns:
            ref = np.nan_to_num(df["ema20"].values, nan=close[0])
        else:
            ref = pd.Series(close).rolling(20).mean().fillna(method="bfill").values

        position = 0.0
        entry_px = 0.0
        equity = []
        trades = []
        cash = self.initial_capital
        bars_in_market = 0

        for i in range(1, len(df)):
            price = close[i]
            regime = str(regimes[i]) if regimes is not None else "sideways"
            sp = self._spacing(i, ref[i], atr)
            tp_rate = sp * self.tp_mult
            sl_rate = max(sp, self.stop_loss_pct)

            # ── EXITS ──
            if position > 0:
                sl_px = entry_px * (1 - sl_rate)
                exit_sl = price <= sl_px
                exit_rsi = regime in ("bull_weak", "sideways") and rsi is not None and rsi[i] > self.rsi_ob
                exit_tp = regime == "bull_strong" and price >= entry_px * (1 + tp_rate)
                exit_regime = regime in ("bear_strong", "bear_weak", "high_vol")
                if exit_sl or exit_rsi or exit_tp or exit_regime:
                    exit_px = price * (1 - self.slippage)
                    proceeds = position * exit_px * (1 - self.commission)
                    pnl = proceeds - (position * entry_px)
                    reason = ("TP" if exit_tp else "SL" if exit_sl else "RSI_OB" if exit_rsi else "REGIME")
                    trades.append({"date": df.index[i], "entry": entry_px, "exit": exit_px,
                                   "pnl": pnl, "pnl_pct": pnl / (position * entry_px) * 100,
                                   "reason": reason, "side": "long"})
                    cash = proceeds
                    position = 0.0
                    entry_px = 0.0
            elif position < 0:
                sl_px = abs(entry_px) * (1 + sl_rate)
                exit_sl = price >= sl_px
                exit_rsi = regime in ("bear_weak", "sideways") and rsi is not None and rsi[i] < self.rsi_os
                exit_tp = regime == "bear_strong" and price <= abs(entry_px) * (1 - tp_rate)
                exit_regime = regime in ("bull_strong", "bull_weak", "high_vol")
                if exit_sl or exit_rsi or exit_tp or exit_regime:
                    units = abs(position)
                    buy_px = price * (1 + self.slippage)
                    buy_cost = units * buy_px * (1 + self.commission)
                    sell_proceeds = units * abs(entry_px) * (1 - self.commission)
                    pnl = sell_proceeds - buy_cost
                    reason = ("TP" if exit_tp else "SL" if exit_sl else "RSI_OS" if exit_rsi else "REGIME")
                    trades.append({"date": df.index[i], "entry": entry_px, "exit": buy_px,
                                   "pnl": pnl, "pnl_pct": pnl / (units * abs(entry_px)) * 100,
                                   "reason": reason, "side": "short"})
                    cash = cash - buy_cost
                    position = 0.0
                    entry_px = 0.0

            # ── ENTRIES ──
            if position == 0 and cash >= self.min_notional:
                if regime == "bull_strong":
                    for level in range(self.n_levels):
                        lv = ref[i] * (1 - sp * (level + 1))
                        if price <= lv:
                            buy_px = price * (1 + self.slippage)
                            units = (cash * (1 - self.commission)) / buy_px
                            position = units
                            cash = 0.0
                            entry_px = buy_px
                            break
                elif regime == "bull_weak" and rsi is not None and rsi[i] < self.rsi_os:
                    buy_px = price * (1 + self.slippage)
                    units = (cash * (1 - self.commission)) / buy_px
                    position = units
                    cash = 0.0
                    entry_px = buy_px
                elif regime == "bear_strong":
                    for level in range(self.n_levels):
                        lv = ref[i] * (1 + sp * (level + 1))
                        if price >= lv:
                            sell_px = price * (1 - self.slippage)
                            units = cash / sell_px
                            position = -units
                            cash += units * sell_px * (1 - self.commission)
                            entry_px = sell_px
                            break
                elif regime == "bear_weak" and rsi is not None and rsi[i] > self.rsi_ob:
                    sell_px = price * (1 - self.slippage)
                    units = cash / sell_px
                    position = -units
                    cash += units * sell_px * (1 - self.commission)
                    entry_px = sell_px
                elif regime == "sideways":
                    if rsi is not None and rsi[i] < self.rsi_os:
                        buy_px = price * (1 + self.slippage)
                        units = (cash * (1 - self.commission)) / buy_px
                        position = units
                        cash = 0.0
                        entry_px = buy_px
                    elif rsi is not None and rsi[i] > self.rsi_ob:
                        sell_px = price * (1 - self.slippage)
                        units = cash / sell_px
                        position = -units
                        cash += units * sell_px * (1 - self.commission)
                        entry_px = sell_px

            if position != 0:
                bars_in_market += 1
            equity.append(cash + abs(position) * price)

        # Cerrar final
        if position > 0:
            lp = df["close"].iloc[-1]
            sp = lp * (1 - self.slippage)
            equity[-1] = position * sp * (1 - self.commission)
        elif position < 0:
            lp = df["close"].iloc[-1]
            bp = lp * (1 + self.slippage)
            units = abs(position)
            equity[-1] = cash - units * bp * (1 + self.commission)

        return self._metrics(equity, trades, bars_in_market)

    def _metrics(self, equity: list, trades: list, bars_in_market: int = 0) -> dict:
        eq = np.array(equity, dtype=float)
        ret = np.diff(eq) / (eq[:-1] + 1e-10)
        total_return = (eq[-1] - self.initial_capital) / self.initial_capital * 100
        final_capital = eq[-1]
        sharpe = (ret.mean() / (ret.std() + 1e-10)) * self.interval_factor if len(ret) > 1 else 0.0
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        max_dd = float(dd.min())
        n_trades = len(trades)
        winners = [t for t in trades if t["pnl"] > 0]
        win_rate = len(winners) / n_trades * 100 if n_trades else 0.0
        gross_profit = sum(t["pnl"] for t in winners) or 0.0
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0)) or 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        trade_pnls = np.array([t["pnl"] for t in trades]) if trades else np.array([0.0])
        sqn = (trade_pnls.mean() / (trade_pnls.std() + 1e-10)) * np.sqrt(n_trades) if n_trades > 1 else 0.0
        dd_sq = np.nan_to_num((dd / 100) ** 2)
        time_in_market_pct = (bars_in_market / len(eq)) * 100 if len(eq) > 1 else 0.0
        return {
            "equity": eq, "drawdowns": dd, "trades": trades,
            "total_return": total_return, "sharpe": sharpe, "max_drawdown": max_dd,
            "ulcer_index": float(np.sqrt(np.mean(dd_sq))),
            "calmar": abs(total_return / max_dd) if max_dd != 0 else 0.0,
            "win_rate": win_rate, "profit_factor": profit_factor, "sqn": sqn,
            "expectancy": float(trade_pnls.mean()), "n_trades": n_trades,
            "final_capital": final_capital, "time_in_market": time_in_market_pct,
        }


# ─────────────────────────────────────────────────────────────
# 3d. PORTFOLIO MANAGER (multi-par × multi-estrategia)
# ─────────────────────────────────────────────────────────────
# 3c. PORTFOLIO MANAGER (multi-par × multi-estrategia)
# ─────────────────────────────────────────────────────────────

class AllocUnit:
    """Unidad de trading: un par × una estrategia × un nivel (para grid)."""

    def __init__(self, uid: str, pair: str, capital: float,
                 engine_class, engine_kwargs: dict,
                 interval_factor: float, df_is: pd.DataFrame, df_oos: pd.DataFrame):
        self.uid = uid
        self.pair = pair
        self.capital = capital
        self.engine_class = engine_class
        self.engine_kwargs = dict(engine_kwargs)
        self.interval_factor = interval_factor
        self.df_is = df_is
        self.df_oos = df_oos
        self.result_is = None
        self.result_oos = None

    def run(self):
        ek = dict(self.engine_kwargs)
        if self.engine_class == BacktestEngine:
            strategy_fn = ek.pop("strategy_fn")
            signals_is = strategy_fn(self.df_is)
            engine_is = BacktestEngine(self.df_is, capital=self.capital,
                                       interval_factor=self.interval_factor, **ek)
            self.result_is = engine_is.run(signals_is)
            signals_oos = strategy_fn(self.df_oos)
            engine_oos = BacktestEngine(self.df_oos, capital=self.capital,
                                        interval_factor=self.interval_factor, **ek)
            self.result_oos = engine_oos.run(signals_oos)
        else:
            engine_is = self.engine_class(self.df_is, capital=self.capital,
                                          interval_factor=self.interval_factor, **ek)
            self.result_is = engine_is.run()
            engine_oos = self.engine_class(self.df_oos, capital=self.capital,
                                           interval_factor=self.interval_factor, **ek)
            self.result_oos = engine_oos.run()
        return self.result_is, self.result_oos

    def score(self):
        r = self.result_is
        if r is None or r["n_trades"] < 5:
            return 0.0
        s = max(0.0, r["sharpe"])
        p = max(0.0, r["profit_factor"])
        return (s * p) ** 0.5 if s > 0 and p > 0 else 0.0

    def metrics(self, phase="oos"):
        r = self.result_oos if phase == "oos" else self.result_is
        if r is None:
            return {}
        keys = ["total_return", "sharpe", "max_drawdown", "win_rate",
                "profit_factor", "n_trades", "final_capital",
                "sqn", "expectancy", "calmar", "ulcer_index", "time_in_market"]
        return {k: r[k] for k in keys}


class PortfolioManager:
    """Portafolio multi-par × multi-estrategia con asignación dinámica post-hoc."""

    def __init__(self, pairs: list, interval: str, days: int,
                 capital: float, commission: float = 0.001,
                 slippage: float = 0.0005, stop_loss_pct: float = 0.0,
                 allow_short: bool = False, min_notional: float = 10.0,
                 grid_pct: float = 0.01, grid_levels: int = 10,
                 min_alloc: float = 0.05, atr_mult: float = 0.0):
        self.pairs = pairs
        self.interval = interval
        self.days = days
        self.capital = capital
        self.commission = commission
        self.slippage = slippage
        self.stop_loss_pct = stop_loss_pct
        self.allow_short = allow_short
        self.min_notional = min_notional
        self.grid_pct = grid_pct
        self.grid_levels = grid_levels
        self.min_alloc = min_alloc
        self.atr_mult = atr_mult
        self.interval_factor = annualization_factor(interval)
        self.units: list[AllocUnit] = []

    def _grid_kwargs(self):
        return dict(commission=self.commission, slippage=self.slippage,
                    grid_pct=self.grid_pct, n_levels=self.grid_levels,
                    min_notional=self.min_notional,
                    stop_loss_pct=self.stop_loss_pct,
                    allow_short=self.allow_short,
                    tp_mult=1.5, atr_mult=self.atr_mult)

    def _strategy_kwargs(self, strategy_fn):
        return dict(commission=self.commission, slippage=self.slippage,
                    stop_loss_pct=self.stop_loss_pct,
                    allow_short=self.allow_short,
                    min_notional=self.min_notional,
                    strategy_fn=strategy_fn)

    def _regime_kwargs(self):
        return dict(commission=self.commission, slippage=self.slippage,
                    stop_loss_pct=self.stop_loss_pct,
                    min_notional=self.min_notional,
                    tp_mult=1.5, atr_mult=2.0,
                    rsi_oversold=30, rsi_overbought=70,
                    n_levels=3)

    def build_units(self):
        strategies = [
            ("Grid",   GridBacktestEngine, self._grid_kwargs(),   False),
            ("RSI",    BacktestEngine,      self._strategy_kwargs(strategy_rsi), True),
            ("Regime", RegimeEngine,        self._regime_kwargs(), False),
        ]
        # Walk-forward: ventanas deslizantes train/test
        train_bars = int(90 * 24 * 60 / self._interval_minutes())  # 90 días
        test_bars  = int(30 * 24 * 60 / self._interval_minutes())  # 30 días
        step_bars  = test_bars
        min_bars   = train_bars + test_bars

        for pair in self.pairs:
            df = add_indicators(fetch_ohlcv(pair, self.interval, self.days))
            df = add_mtf_trend(df, pair, self.days)
            df = add_regime(df)
            n = len(df)
            if n < min_bars:
                print(f"{YELLOW}⚠  {pair}: solo {n} velas, mínimo {min_bars} para walk-forward{RESET}")
                continue
            n_windows = 0
            for start in range(0, n - min_bars + 1, step_bars):
                df_is = df.iloc[start:start + train_bars]
                df_oos = df.iloc[start + train_bars:start + min_bars]
                # Recalcular interval_factor para este window (OOS puede tener distinta longitud)
                for name, engine_cls, kwargs, _ in strategies:
                    w_id = f"{name}:{pair}:W{n_windows}"
                    self.units.append(AllocUnit(
                        w_id, pair, 0,
                        engine_cls, dict(kwargs),
                        self.interval_factor, df_is, df_oos))
                n_windows += 1
            print(f"{CYAN}📊  {pair}: {n_windows} ventanas walk-forward ({train_bars} IS + {test_bars} OOS){RESET}")

    def _interval_minutes(self) -> int:
        return {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
                "1h": 60, "2h": 120, "4h": 240, "1d": 1440}.get(self.interval, 15)

    def run(self):
        self.build_units()
        n = len(self.units)
        cap_per = self.capital / n
        for u in self.units:
            u.capital = cap_per
        for u in self.units:
            print(f"  {CYAN}▶{RESET}  {u.uid}")
            u.run()
        # Agrupar por (estrategia, par) y promediar scores
        groups = {}
        for u in self.units:
            key = ":".join(u.uid.split(":")[:2])
            if key not in groups:
                groups[key] = []
            groups[key].append(u)
        scores = {}
        for key, units in groups.items():
            window_scores = [u.score() for u in units]
            scores[key] = np.mean(window_scores) if window_scores else 0.0
        total_s = sum(scores.values())
        if total_s > 0:
            weights = {k: v / total_s for k, v in scores.items()}
        else:
            weights = {k: 1.0 / len(scores) for k in scores}
        weights = {k: max(self.min_alloc, w) for k, w in weights.items()}
        w_sum = sum(weights.values())
        for k in weights:
            weights[k] /= w_sum
        for u in self.units:
            key = ":".join(u.uid.split(":")[:2])
            if u.result_oos is not None:
                u.result_oos["alloc_pct"] = weights.get(key, 0) * 100
                u.result_is["alloc_pct"] = weights.get(key, 0) * 100
        self._groups = groups
        self._weights = weights

    def print_results(self):
        print(f"\n{'═'*140}")
        print(f"{BOLD}  WALK-FORWARD PORTFOLIO  (capital: ${self.capital:.0f} | pares: {', '.join(self.pairs)}){RESET}")
        print(f"{'═'*140}\n")
        headers = ["Unidad", "Wins", "IS:Sh∅", "IS:PF∅", "IS:Tr∑",
                   "OOS:R%∅", "OOS:Sh∅", "OOS:DD∅", "OOS:PF∅", "OOS:Tr∑",
                   "Alloc%"]
        rows = []
        for key in sorted(self._weights):
            units = self._groups[key]
            is_scores = [u.score() for u in units if u.score() > 0]
            n_wins = len(is_scores)
            is_sharpe = np.mean([u.result_is["sharpe"] for u in units if u.result_is]) if units else 0
            is_pf = np.mean([u.result_is["profit_factor"] for u in units if u.result_is]) if units else 0
            is_tr = sum(u.result_is["n_trades"] for u in units if u.result_is)
            oos_sharpe = np.mean([u.result_oos["sharpe"] for u in units if u.result_oos]) if units else 0
            oos_pf = np.mean([u.result_oos["profit_factor"] for u in units if u.result_oos]) if units else 0
            oos_ret = np.mean([u.result_oos["total_return"] for u in units if u.result_oos]) if units else 0
            oos_dd = np.mean([u.result_oos["max_drawdown"] for u in units if u.result_oos]) if units else 0
            oos_tr = sum(u.result_oos["n_trades"] for u in units if u.result_oos)
            c = GREEN if oos_sharpe >= 0 else RED
            rows.append([
                key,
                f"{n_wins}/{len(units)}",
                f"{is_sharpe:.2f}",
                f"{is_pf:.2f}",
                str(is_tr),
                f"{c}{oos_ret:+.1f}%{RESET}",
                f"{c}{oos_sharpe:.2f}{RESET}",
                f"{RED}{oos_dd:.1f}%{RESET}",
                f"{oos_pf:.2f}",
                str(oos_tr),
                f"{self._weights.get(key, 0)*100:.1f}%",
            ])
        if HAS_TABULATE:
            print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
        else:
            print("  ".join(h.ljust(12) for h in headers))
            print("-" * 170)
            for row in rows:
                print("  ".join(str(c).ljust(12) for c in row))
        # Portfolio OOS combinado
        total_oos_tr = sum(u.result_oos["n_trades"] for u in self.units if u.result_oos)
        total_is_tr = sum(u.result_is["n_trades"] for u in self.units if u.result_is)
        print(f"\n  {BOLD}Trades totales:{RESET} IS={total_is_tr}  OOS={total_oos_tr}  |  {BOLD}Unidades:{RESET} {len(self.units)}")
        print(f"{YELLOW}📌  Walk-forward: media de Scores (sqrt(Sh×PF)) en todas las ventanas.{RESET}")
        print(f"{YELLOW}   Alloc% ∝ media del Score. Piso {self.min_alloc:.0%}. Wins = ventanas con score > 0{RESET}")

    def save_state(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = f"{CACHE_DIR}/portfolio_{'_'.join(p.replace('/','_') for p in self.pairs)}_{self.interval}_{self.days}d.pkl"
        data = {"pairs": self.pairs, "interval": self.interval, "days": self.days,
                "capital": self.capital, "min_alloc": self.min_alloc}
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  {CYAN}💾  Portfolio state saved{RESET}")


# ─────────────────────────────────────────────────────────────
# 4. ESTRATEGIAS
# ─────────────────────────────────────────────────────────────

def _make_signals(arr: np.ndarray, index) -> pd.Series:
    """Helper: convierte array numpy de señales a Series con índice del df."""
    return pd.Series(arr, index=index, dtype=int)


def strategy_grid(df: pd.DataFrame, grid_pct: float = 0.03) -> pd.Series:
    """
    Grid Trading con niveles fijos.
    Compra cuando el precio cae grid_pct% desde el nivel de referencia.
    Vende cuando sube grid_pct% desde el precio de entrada.
    El nivel de referencia se actualiza al inicio y tras cada ciclo completo.
    """
    close    = df["close"].values
    arr      = np.zeros(len(df), dtype=int)
    in_trade = False
    ref      = close[0]
    entry    = 0.0

    for i in range(1, len(df)):
        price = close[i]
        if not in_trade:
            if price < ref * (1 - grid_pct):
                arr[i]   = 1
                in_trade = True
                entry    = price
        else:
            if price > entry * (1 + grid_pct):
                arr[i]   = -1
                in_trade = False
                ref      = price
    return _make_signals(arr, df.index)


def strategy_rsi(df: pd.DataFrame, oversold: float = 32, overbought: float = 68) -> pd.Series:
    """RSI Mean Reversion: compra en sobreventa, vende en sobrecompra."""
    rsi_vals = df["rsi"].values
    arr      = np.zeros(len(df), dtype=int)
    in_trade = False

    for i in range(1, len(df)):
        r = rsi_vals[i]
        if not in_trade and r < oversold:
            arr[i]   = 1
            in_trade = True
        elif in_trade and r > overbought:
            arr[i]   = -1
            in_trade = False
    return _make_signals(arr, df.index)


def strategy_ema_cross(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """EMA Crossover: compra en golden cross EMA 12/26, vende en death cross."""
    ema_f    = df["ema12"].values
    ema_s    = df["ema26"].values
    arr      = np.zeros(len(df), dtype=int)
    in_trade = False

    for i in range(1, len(df)):
        golden_cross = ema_f[i] > ema_s[i] and ema_f[i-1] <= ema_s[i-1]
        death_cross  = ema_f[i] < ema_s[i] and ema_f[i-1] >= ema_s[i-1]

        if not in_trade and golden_cross:
            arr[i]   = 1
            in_trade = True
        elif in_trade and death_cross:
            arr[i]   = -1
            in_trade = False
    return _make_signals(arr, df.index)


def strategy_rsi5_mr(df: pd.DataFrame, oversold: float = 40, overbought: float = 60) -> pd.Series:
    """RSI5 Mean Reversion rápida: compra en RSI5 bajo, vende en RSI5 alto."""
    rsi5 = df["rsi5"].values
    arr = np.zeros(len(df), dtype=int)
    in_trade = False
    for i in range(1, len(df)):
        r = rsi5[i]
        if not in_trade and r < oversold:
            arr[i] = 1
            in_trade = True
        elif in_trade and r > overbought:
            arr[i] = -1
            in_trade = False
    return _make_signals(arr, df.index)


def strategy_bollinger(df: pd.DataFrame, std: float = 2.0) -> pd.Series:
    """Bollinger Bands: compra en banda inferior, vende en banda superior."""
    close    = df["close"].values
    upper    = df["bb_upper"].values
    lower    = df["bb_lower"].values
    arr      = np.zeros(len(df), dtype=int)
    in_trade = False

    for i in range(1, len(df)):
        if not in_trade and close[i] < lower[i]:
            arr[i]   = 1
            in_trade = True
        elif in_trade and close[i] > upper[i]:
            arr[i]   = -1
            in_trade = False
    return _make_signals(arr, df.index)


# ── NUEVAS ESTRATEGIAS ──

def strategy_scalp_momentum(df: pd.DataFrame, fast: int = 5, slow: int = 12, rsi_period: int = 5,
                            rsi_entry: float = 50, rsi_exit: float = 40) -> pd.Series:
    """Scalping momentum: EMA cross + RSI > entry confirma momentum (no contrarian)."""
    ema_f = df[f"ema{fast}"].values
    ema_s = df[f"ema{slow}"].values
    rsi   = df[f"rsi{rsi_period}"].values
    arr   = np.zeros(len(df), dtype=int)
    in_trade = False

    for i in range(1, len(df)):
        gc = ema_f[i] > ema_s[i] and ema_f[i - 1] <= ema_s[i - 1]
        dc = ema_f[i] < ema_s[i] and ema_f[i - 1] >= ema_s[i - 1]
        if not in_trade and gc and rsi[i] > rsi_entry:
            arr[i] = 1
            in_trade = True
        elif in_trade and (dc or rsi[i] < rsi_exit):
            arr[i] = -1
            in_trade = False
    return _make_signals(arr, df.index)


def strategy_vwap_reversion(df: pd.DataFrame, n_std: float = 2.0) -> pd.Series:
    """VWAP reversion: compra debajo de VWAP - n_std, vende encima de VWAP + n_std."""
    close = df["close"].values
    vw_lower = df["vwap_lower"].values
    vw_upper = df["vwap_upper"].values
    arr = np.zeros(len(df), dtype=int)
    in_trade = False

    for i in range(1, len(df)):
        if not in_trade and close[i] < vw_lower[i]:
            arr[i] = 1
            in_trade = True
        elif in_trade and close[i] > vw_upper[i]:
            arr[i] = -1
            in_trade = False
    return _make_signals(arr, df.index)


def strategy_volume_breakout(df: pd.DataFrame, vol_mult: float = 2.0, atr_mult: float = 1.0) -> pd.Series:
    """Volume breakout: compra cuando volumen>>media + precio rompe + momentum."""
    close = df["close"].values
    vol = df["volume"].values
    vol_ma = df["vol_ma20"].values
    atr = df["atr"].values
    ema20 = df["ema20"].values
    arr = np.zeros(len(df), dtype=int)
    in_trade = False

    for i in range(1, len(df)):
        if not in_trade:
            if vol[i] > vol_ma[i] * vol_mult and abs(close[i] - close[i - 1]) > atr[i] * atr_mult:
                if close[i] > ema20[i]:  # breakout alcista
                    arr[i] = 1
                    in_trade = True
                elif close[i] < ema20[i]:  # breakout bajista
                    arr[i] = -1
                    in_trade = True
        else:
            # Salir cuando el momentum se agota: vela con volumen bajo o precio vuelve a EMA
            if vol[i] < vol_ma[i] * 0.5 or abs(close[i] - ema20[i]) / ema20[i] < 0.001:
                arr[i] = -1 if close[i] > ema20[i] else 1
                in_trade = False
    return _make_signals(arr, df.index)


# ─────────────────────────────────────────────────────────────
# 4b. FILTRO DE SEÑALES
# ─────────────────────────────────────────────────────────────

def filter_signals(df: pd.DataFrame, signals: pd.Series, vol_factor: float = 0.0) -> pd.Series:
    """Anula señales cuando el volumen está por debajo de su media móvil."""
    if vol_factor <= 0:
        return signals
    vol_ma = df["volume"].rolling(20).mean()
    mask = df["volume"] < vol_ma * vol_factor
    return signals.where(~mask, 0)


# ─────────────────────────────────────────────────────────────
# 5. OPTIMIZACIÓN DE PARÁMETROS (grid search simple)
# ─────────────────────────────────────────────────────────────

def optimize_rsi(df_train: pd.DataFrame, capital: float, interval_factor: float = np.sqrt(252),
                 slippage: float = 0.0005, stop_loss_pct: float = 0.0,
                 allow_short: bool = False, vol_factor: float = 0.0,
                 min_notional: float = 10.0) -> dict:
    """Busca los mejores parámetros RSI sobre datos de entrenamiento."""
    print(f"\n{CYAN}🔍 Optimizando RSI sobre 70% in-sample...{RESET}")
    best = {"sharpe": -999}

    oversold_range   = range(25, 40, 3)
    overbought_range = range(60, 78, 3)

    engine = BacktestEngine(df_train, capital, interval_factor=interval_factor,
                            slippage=slippage, stop_loss_pct=stop_loss_pct,
                            allow_short=allow_short, min_notional=min_notional)
    results = []

    for os in oversold_range:
        for ob in overbought_range:
            if os >= ob:
                continue
            signals = strategy_rsi(df_train, oversold=os, overbought=ob)
            if vol_factor > 0:
                signals = filter_signals(df_train, signals, vol_factor)
            res = engine.run(signals)
            results.append({"oversold": os, "overbought": ob, **res})
            if res["sharpe"] > best["sharpe"]:
                best = {"oversold": os, "overbought": ob, **res}

    print(f"{GREEN}✔  Mejor combinación: oversold={best['oversold']} overbought={best['overbought']}")
    print(f"   Sharpe={best['sharpe']:.2f}  Return={best['total_return']:.1f}%{RESET}")
    return best


def optimize_ema(df_train: pd.DataFrame, capital: float, interval_factor: float = np.sqrt(252),
                 slippage: float = 0.0005, stop_loss_pct: float = 0.0,
                 allow_short: bool = False, vol_factor: float = 0.0,
                 min_notional: float = 10.0) -> dict:
    """Busca los mejores períodos de EMA sobre datos de entrenamiento."""
    print(f"\n{CYAN}🔍 Optimizando EMA sobre 70% in-sample...{RESET}")
    best = {"sharpe": -999}
    engine = BacktestEngine(df_train, capital, interval_factor=interval_factor,
                            slippage=slippage, stop_loss_pct=stop_loss_pct,
                            allow_short=allow_short, min_notional=min_notional)
    for fast in range(5, 30, 5):
        for slow in range(fast + 10, 60, 10):
            if slow <= fast:
                continue
            signals = strategy_ema_cross(df_train, fast=fast, slow=slow)
            if vol_factor > 0:
                signals = filter_signals(df_train, signals, vol_factor)
            res = engine.run(signals)
            if res["sharpe"] > best["sharpe"]:
                best = {"fast": fast, "slow": slow, **res}
    print(f"{GREEN}✔  Mejor combinación: fast={best['fast']} slow={best['slow']}")
    print(f"   Sharpe={best['sharpe']:.2f}  Return={best['total_return']:.1f}%{RESET}")
    return best


def optimize_bollinger(df_train: pd.DataFrame, capital: float, interval_factor: float = np.sqrt(252),
                       slippage: float = 0.0005, stop_loss_pct: float = 0.0,
                       allow_short: bool = False, vol_factor: float = 0.0,
                       min_notional: float = 10.0) -> dict:
    """Busca el mejor ancho de bandas de Bollinger sobre datos de entrenamiento."""
    print(f"\n{CYAN}🔍 Optimizando Bollinger sobre 70% in-sample...{RESET}")
    best = {"sharpe": -999}
    engine = BacktestEngine(df_train, capital, interval_factor=interval_factor,
                            slippage=slippage, stop_loss_pct=stop_loss_pct,
                            allow_short=allow_short, min_notional=min_notional)
    for std in [1.5, 2.0, 2.5, 3.0]:
        signals = strategy_bollinger(df_train, std=std)
        if vol_factor > 0:
            signals = filter_signals(df_train, signals, vol_factor)
        res = engine.run(signals)
        if res["sharpe"] > best["sharpe"]:
            best = {"std": std, **res}
    print(f"{GREEN}✔  Mejor combinación: std={best['std']}")
    print(f"   Sharpe={best['sharpe']:.2f}  Return={best['total_return']:.1f}%{RESET}")
    return best


def optimize_grid(df_train: pd.DataFrame, capital: float, interval_factor: float = np.sqrt(252),
                  slippage: float = 0.0005, stop_loss_pct: float = 0.0,
                  allow_short: bool = False, vol_factor: float = 0.0,
                  min_notional: float = 10.0) -> dict:
    """Busca el mejor grid_pct sobre datos de entrenamiento."""
    print(f"\n{CYAN}🔍 Optimizando Grid sobre 70% in-sample...{RESET}")
    best = {"sharpe": -999}
    engine = BacktestEngine(df_train, capital, interval_factor=interval_factor,
                            slippage=slippage, stop_loss_pct=stop_loss_pct,
                            allow_short=allow_short, min_notional=min_notional)
    for pct in [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10]:
        signals = strategy_grid(df_train, grid_pct=pct)
        if vol_factor > 0:
            signals = filter_signals(df_train, signals, vol_factor)
        res = engine.run(signals)
        if res["sharpe"] > best["sharpe"]:
            best = {"grid_pct": pct, **res}
    print(f"{GREEN}✔  Mejor combinación: grid_pct={best['grid_pct']}")
    print(f"   Sharpe={best['sharpe']:.2f}  Return={best['total_return']:.1f}%{RESET}")
    return best


def optimize_grid_multi(df_train: pd.DataFrame, capital: float, interval_factor: float = np.sqrt(252),
                        slippage: float = 0.0005, stop_loss_pct: float = 0.0,
                        allow_short: bool = False, vol_factor: float = 0.0,
                        min_notional: float = 10.0, grid_levels: int = 10) -> dict:
    """Busca el mejor grid_pct para Grid multinivel sobre datos de entrenamiento."""
    print(f"\n{CYAN}🔍 Optimizando Grid Multinivel sobre 70% in-sample...{RESET}")
    best = {"sharpe": -999}
    for pct in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]:
        engine = GridBacktestEngine(df_train, capital=capital, commission=0.001,
                                    slippage=slippage, grid_pct=pct, n_levels=grid_levels,
                                    interval_factor=interval_factor, min_notional=min_notional,
                                    stop_loss_pct=stop_loss_pct)
        res = engine.run()
        if res["sharpe"] > best["sharpe"]:
            best = {"grid_pct": pct, **res}
    print(f"{GREEN}✔  Mejor combinación: grid_pct={best['grid_pct']}")
    print(f"   Sharpe={best['sharpe']:.2f}  Return={best['total_return']:.1f}%{RESET}")
    return best


# ─────────────────────────────────────────────────────────────
# 5b. BARRIO PARAMÉTRICO (sweep) con validación IS/OOS
# ─────────────────────────────────────────────────────────────

def sweep_grid_params(df: pd.DataFrame, capital: float = 1000.0,
                      interval_factor: float = np.sqrt(252 * 96),
                      slippage: float = 0.0005, stop_loss_pct: float = 0.0,
                      allow_short: bool = False, min_notional: float = 10.0,
                      n_levels: int = 10) -> list[dict]:
    """Sweep grid_pct × tp_mult (y atr_mult) sobre 70% IS, valida 30% OOS."""
    split = int(len(df) * 0.70)
    df_is = df.iloc[:split]
    df_oos = df.iloc[split:]
    results = []
    params = []
    for gp in [0.005, 0.01, 0.02, 0.03]:
        for tp in [1.0, 1.5, 2.0, 3.0]:
            params.append(dict(grid_pct=gp, tp_mult=tp, atr_mult=0.0))
    for am in [0.5, 1.0, 2.0, 3.0]:
        for tp in [1.0, 1.5, 2.0]:
            params.append(dict(grid_pct=0.01, tp_mult=tp, atr_mult=am))

    for kw in params:
        try:
            engine = GridBacktestEngine(df_is, capital=capital, commission=0.001,
                                        slippage=slippage, grid_pct=kw["grid_pct"],
                                        n_levels=n_levels, interval_factor=interval_factor,
                                        min_notional=min_notional, stop_loss_pct=stop_loss_pct,
                                        allow_short=allow_short,
                                        tp_mult=kw["tp_mult"], atr_mult=kw["atr_mult"])
            res_is = engine.run()
            engine_oos = GridBacktestEngine(df_oos, capital=capital, commission=0.001,
                                            slippage=slippage, grid_pct=kw["grid_pct"],
                                            n_levels=n_levels, interval_factor=interval_factor,
                                            min_notional=min_notional, stop_loss_pct=stop_loss_pct,
                                            allow_short=allow_short,
                                            tp_mult=kw["tp_mult"], atr_mult=kw["atr_mult"])
            res_oos = engine_oos.run()
            if res_is["n_trades"] >= 5:
                results.append({**kw, "is_sharpe": res_is["sharpe"], "is_pf": res_is["profit_factor"],
                                "is_ret": res_is["total_return"], "is_tr": res_is["n_trades"],
                                "oos_sharpe": res_oos["sharpe"], "oos_pf": res_oos["profit_factor"],
                                "oos_ret": res_oos["total_return"], "oos_tr": res_oos["n_trades"]})
        except Exception:
            continue

    results.sort(key=lambda r: r["is_sharpe"], reverse=True)
    return results


def sweep_rsi_params(df: pd.DataFrame, capital: float = 1000.0,
                     interval_factor: float = np.sqrt(252 * 96),
                     slippage: float = 0.0005, stop_loss_pct: float = 0.0,
                     allow_short: bool = False, min_notional: float = 10.0) -> list[dict]:
    """Sweep oversold × overbought para RSI sobre 70% IS, valida 30% OOS."""
    split = int(len(df) * 0.70)
    df_is = df.iloc[:split]
    df_oos = df.iloc[split:]
    results = []
    for os in [20, 25, 30, 35, 40]:
        for ob in [60, 65, 70, 75, 80]:
            if ob - os < 20:
                continue
            try:
                signals_is = strategy_rsi(df_is, oversold=os, overbought=ob)
                engine = BacktestEngine(df_is, capital=capital, commission=0.001,
                                        slippage=slippage, stop_loss_pct=stop_loss_pct,
                                        allow_short=allow_short, interval_factor=interval_factor,
                                        min_notional=min_notional)
                res_is = engine.run(signals_is)
                signals_oos = strategy_rsi(df_oos, oversold=os, overbought=ob)
                engine_oos = BacktestEngine(df_oos, capital=capital, commission=0.001,
                                            slippage=slippage, stop_loss_pct=stop_loss_pct,
                                            allow_short=allow_short,
                                            interval_factor=interval_factor,
                                            min_notional=min_notional)
                res_oos = engine_oos.run(signals_oos)
                if res_is["n_trades"] >= 5:
                    results.append({"oversold": os, "overbought": ob,
                                    "is_sharpe": res_is["sharpe"], "is_pf": res_is["profit_factor"],
                                    "is_ret": res_is["total_return"], "is_tr": res_is["n_trades"],
                                    "oos_sharpe": res_oos["sharpe"], "oos_pf": res_oos["profit_factor"],
                                    "oos_ret": res_oos["total_return"], "oos_tr": res_oos["n_trades"]})
            except Exception:
                continue
    results.sort(key=lambda r: r["is_sharpe"], reverse=True)
    return results


def print_sweep(results: list[dict], title: str):
    """Imprime tabla de sweep paramétrico."""
    if not results:
        print(f"{YELLOW}⚠  Sin resultados válidos (≥5 trades){RESET}")
        return
    print(f"\n{CYAN}📊 {title}{RESET}")
    print(f"{'─'*140}")
    headers = ["Params", "IS:Sh", "IS:PF", "IS:Tr", "OOS:Sh", "OOS:PF", "OOS:R%", "OOS:Tr"]
    rows = []
    for r in results[:20]:
        label = ", ".join(f"{k}={v}" for k, v in r.items()
                          if k not in ("is_sharpe", "is_pf", "is_ret", "is_tr",
                                       "oos_sharpe", "oos_pf", "oos_ret", "oos_tr"))
        c = GREEN if r["oos_sharpe"] >= 0 else RED
        rows.append([
            label,
            f"{r['is_sharpe']:.2f}",
            f"{r['is_pf']:.2f}",
            str(r["is_tr"]),
            f"{c}{r['oos_sharpe']:.2f}{RESET}",
            f"{r['oos_pf']:.2f}",
            f"{c}{r['oos_ret']:+.1f}%{RESET}",
            str(r["oos_tr"]),
        ])
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        for row in rows:
            print("  ".join(str(c).ljust(14) for c in row))
    print(f"{'─'*140}")


OPTIMIZERS = {
    "rsi": optimize_rsi,
    "ema": optimize_ema,
    "bollinger": optimize_bollinger,
    "grid": optimize_grid,
    "grid_multi": optimize_grid_multi,
}


# ─────────────────────────────────────────────────────────────
# 6. PRESENTACIÓN DE RESULTADOS
# ─────────────────────────────────────────────────────────────

def _row(name: str, r: dict, best_sharpe: float):
    ret_color = GREEN if r["total_return"] >= 0 else RED
    is_best   = "⭐" if r["sharpe"] == best_sharpe else "  "
    return [
        f"{is_best} {name}",
        f"{ret_color}{r['total_return']:+.1f}%{RESET}",
        f"{r['sharpe']:.2f}",
        f"{RED}{r['max_drawdown']:.1f}%{RESET}",
        f"{r['win_rate']:.0f}%",
        f"{r['profit_factor']:.2f}",
        f"{r['n_trades']}",
        f"${r['final_capital']:.0f}",
    ]

def print_results_cmp(results_is: dict, results_oos: dict, capital: float):
    """Imprime tabla comparativa IS vs OOS."""
    print(f"\n{'═'*110}")
    print(f"{BOLD}  RESULTADOS DEL BACKTEST  (capital inicial: ${capital:.0f}){RESET}")
    print(f"{'═'*110}\n")

    all_sharpes = list(results_is.values()) + list(results_oos.values())
    best_sharpe = max(r["sharpe"] for r in all_sharpes)
    headers = ["Estrategia", "Ret IS", "Sh IS", "DD IS",
               "Ret OOS", "Sh OOS", "DD OOS", "W%", "PF", "Trades", "Sig.Dens", "T.Market"]
    rows = []
    for name in results_is:
        is_r = results_is[name]
        oos_r = results_oos[name]
        n_bars = len(is_r["equity"])
        sig_dens = is_r["n_trades"] / n_bars * 100 if n_bars else 0
        rows.append([
            name,
            f"{GREEN if is_r['total_return']>=0 else RED}{is_r['total_return']:+.1f}%{RESET}",
            f"{is_r['sharpe']:.2f}",
            f"{RED}{is_r['max_drawdown']:.1f}%{RESET}",
            f"{GREEN if oos_r['total_return']>=0 else RED}{oos_r['total_return']:+.1f}%{RESET}",
            f"{oos_r['sharpe']:.2f}",
            f"{RED}{oos_r['max_drawdown']:.1f}%{RESET}",
            f"{is_r['win_rate']:.0f}%",
            f"{is_r['profit_factor']:.2f}",
            str(is_r['n_trades']),
            f"{sig_dens:.1f}%",
            f"{is_r['time_in_market']:.0f}%",
        ])

    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        print("  ".join(h.ljust(12) for h in headers))
        print("-" * 130)
        for row in rows:
            print("  ".join(str(c).ljust(12) for c in row))

    print(f"\n{YELLOW}📌  Sharpe > 1.0 · PF > 1.5 · DD < 20% · Sig.Dens = trades/velas · T.Market = tiempo en posición{RESET}")


def print_trade_summary(name: str, trades: list, n: int = 5):
    """Muestra las últimas N operaciones de una estrategia."""
    if not trades:
        print(f"\n{YELLOW}  {name}: sin operaciones cerradas.{RESET}")
        return

    print(f"\n{CYAN}  Últimas {min(n, len(trades))} operaciones — {name}:{RESET}")
    for t in trades[-n:]:
        color = GREEN if t["pnl"] > 0 else RED
        print(f"    {t['date'].strftime('%Y-%m-%d')}  "
              f"entrada ${t['entry']:.2f}  salida ${t['exit']:.2f}  "
              f"{color}PnL: {t['pnl']:+.2f} ({t['pnl_pct']:+.1f}%){RESET}")


def export_csv(results: dict, pair: str, trade_log: bool = False):
    """Exporta equity curves y trades a CSV."""
    base = pair.replace("/", "_")
    dfs = {}
    all_trades = []
    for key, val in results.items():
        if isinstance(val, dict) and "equity" not in val:
            for name, r in val.items():
                dfs[f"{key}_{name}"] = r["equity"]
                if trade_log and r["trades"]:
                    for t in r["trades"]:
                        all_trades.append({"split": key, "strategy": name, **t})
        else:
            dfs[key] = val["equity"]

    eq_file = f"equity_{base}.csv"
    pd.DataFrame(dfs).to_csv(eq_file)
    print(f"\n{GREEN}💾 Equity curves exportadas a: {eq_file}{RESET}")

    if trade_log and all_trades:
        tr_file = f"trades_{base}.csv"
        pd.DataFrame(all_trades).to_csv(tr_file, index=False)
        print(f"{GREEN}📋  Trade log exportado a: {tr_file}{RESET}")


# ─────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Crypto Backtest Framework")
    parser.add_argument("--pair",      default="BTC/USDT", help="Par de trading (default: BTC/USDT)")
    parser.add_argument("--interval",  default="1d",        help="Intervalo de velas: 1h, 4h, 1d (default: 1d)")
    parser.add_argument("--days",      type=int, default=365, help="Días históricos (default: 365)")
    parser.add_argument("--capital",   type=float, default=500, help="Capital inicial en USDT (default: 500)")
    parser.add_argument("--commission", type=float, default=0.001, help="Comisión por operación (default: 0.001)")
    parser.add_argument("--slippage",  type=float, default=0.0005, help="Slippage por operación (default: 0.0005)")
    parser.add_argument("--stop-loss", type=float, default=0.0, help="Stop-loss %% (default: 0 = desactivado)")
    parser.add_argument("--allow-short", action="store_true", help="Permite posiciones cortas")
    parser.add_argument("--vol-filter", type=float, default=0.0, help="Filtro volumen: anula señales si vol < media*N (default: 0 = off)")
    parser.add_argument("--optimize",  action="store_true",  help="Optimiza parámetros de estrategias")
    parser.add_argument("--optimize-strategy", default="rsi", help="Estrategia a optimizar: rsi | ema | bollinger | grid | grid_multi (default: rsi)")
    parser.add_argument("--export",    action="store_true",  help="Exporta equity curves a CSV")
    parser.add_argument("--trade-log", action="store_true",  help="Exporta trades individuales a CSV (requiere --export)")
    parser.add_argument("--min-notional", type=float, default=10.0, help="Mínimo nocional por orden en USDT (default: 10)")
    parser.add_argument("--grid-pct",   type=float, default=0.01, help="Grid %% para grid multinivel (default: 0.01)")
    parser.add_argument("--grid-levels", type=int,   default=10,   help="Niveles de la grid multinivel (default: 10)")
    parser.add_argument("--log-file",  default="",           help="Guarda el output completo en un archivo de log")
    parser.add_argument("--multi-pair", default="",          help="Modo portafolio: pares separados por coma (ej: BTC/USDT,ETH/USDT,SOL/USDT)")
    parser.add_argument("--min-alloc", type=float, default=0.05, help="Piso mínimo de asignación por unidad (default: 0.05)")
    parser.add_argument("--grid-atr-mult", type=float, default=0.0, help="Multiplicador ATR para grid dinámico (0=fijo) (default: 0)")
    parser.add_argument("--sweep", default="", help="Barrido paramétrico: grid | rsi (ej: --sweep rsi)")
    args = parser.parse_args()

    if args.log_file:
        logging.basicConfig(filename=args.log_file, level=logging.INFO,
                            format="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        logging.info("=== BACKTEST STARTED ===")
        logging.info("Args: %s", vars(args))

    print(f"\n{BOLD}{CYAN}{'═'*80}")
    print(f"  CRYPTO BACKTEST FRAMEWORK")
    print(f"  Par: {args.pair}  |  Intervalo: {args.interval}  |  Días: {args.days}")
    print(f"  Capital: ${args.capital:.0f}  |  Comisión: {args.commission:.1%}  |  Slippage: {args.slippage:.2%}")
    print(f"  Stop-loss: {'off' if args.stop_loss == 0 else f'{args.stop_loss*100:.0f}%'}  |  Short: {'on' if args.allow_short else 'off'}")
    print(f"{'═'*80}{RESET}\n")

    # ── MODO SWEEP (barrido paramétrico) ──
    if args.sweep:
        df_raw = fetch_ohlcv(args.pair, args.interval, args.days)
        df = add_indicators(df_raw)
        interval_factor = annualization_factor(args.interval)
        if args.sweep == "rsi":
            results = sweep_rsi_params(df, capital=args.capital,
                                       interval_factor=interval_factor,
                                       slippage=args.slippage, stop_loss_pct=args.stop_loss,
                                       allow_short=args.allow_short, min_notional=args.min_notional)
            print_sweep(results, f"RSI Param Sweep — {args.pair} ({args.interval}, {args.days}d)")
        elif args.sweep == "grid":
            results = sweep_grid_params(df, capital=args.capital,
                                        interval_factor=interval_factor,
                                        slippage=args.slippage, stop_loss_pct=args.stop_loss,
                                        allow_short=args.allow_short, min_notional=args.min_notional,
                                        n_levels=args.grid_levels)
            print_sweep(results, f"Grid Param Sweep — {args.pair} ({args.interval}, {args.days}d)")
        else:
            print(f"{RED}--sweep debe ser 'grid' o 'rsi'{RESET}")
        return

    # ── MODO PORTFOLIO (multi-par) ──
    if args.multi_pair:
        pairs = [p.strip() for p in args.multi_pair.split(",") if p.strip()]
        pm = PortfolioManager(
            pairs, args.interval, args.days, args.capital,
            commission=args.commission, slippage=args.slippage,
            stop_loss_pct=args.stop_loss, allow_short=args.allow_short,
            min_notional=args.min_notional,
            grid_pct=args.grid_pct, grid_levels=args.grid_levels,
            min_alloc=args.min_alloc,
            atr_mult=args.grid_atr_mult,
        )
        pm.run()
        pm.print_results()
        pm.save_state()
        if args.export:
            print(f"\n{YELLOW}⚠  Export CSV no disponible en modo portfolio (próxima versión){RESET}")
        return

    # ── MODO SINGLE-PAIR ──
    df_raw = fetch_ohlcv(args.pair, args.interval, args.days)
    df = add_indicators(df_raw)
    print(f"  Velas útiles (post-indicadores): {len(df)}\n")

    split = int(len(df) * 0.70)
    df_train = df.iloc[:split]
    df_test  = df.iloc[split:]
    print(f"  In-sample:      {len(df_train)} velas  ({df_train.index[0].date()} → {df_train.index[-1].date()})")
    print(f"  Out-of-sample:  {len(df_test)} velas  ({df_test.index[0].date()} → {df_test.index[-1].date()})\n")

    interval_factor = annualization_factor(args.interval)
    engine_kwargs = dict(
        capital=args.capital, commission=args.commission, slippage=args.slippage,
        stop_loss_pct=args.stop_loss, allow_short=args.allow_short,
        interval_factor=interval_factor, min_notional=args.min_notional,
    )

    strategy_fns = [
        ("Grid Trading",    strategy_grid),
        ("RSI Mean Rev.",   strategy_rsi),
        ("RSI5 Mean Rev.",  strategy_rsi5_mr),
        ("EMA Crossover",   strategy_ema_cross),
        ("Bollinger Bands", strategy_bollinger),
        ("Scalp Momentum",  strategy_scalp_momentum),
        ("VWAP Reversion",  strategy_vwap_reversion),
        ("Volume Breakout", strategy_volume_breakout),
    ]

    print(f"{CYAN}⚙  Corriendo backtest (IS y OOS) para {len(strategy_fns)} estrategias...{RESET}")
    results_is  = {}
    results_oos = {}
    for name, fn in strategy_fns:
        sig_is  = fn(df_train)
        sig_oos = fn(df_test)
        if args.vol_filter > 0:
            sig_is  = filter_signals(df_train, sig_is, args.vol_filter)
            sig_oos = filter_signals(df_test, sig_oos, args.vol_filter)
        engine_is  = BacktestEngine(df_train, **engine_kwargs)
        engine_oos = BacktestEngine(df_test,  **engine_kwargs)
        results_is[name]  = engine_is.run(sig_is)
        results_oos[name] = engine_oos.run(sig_oos)
        print(f"  {GREEN}✔{RESET}  {name}")

    print(f"\n{CYAN}⚙  Corriendo Grid Multinivel (IS y OOS)...{RESET}")
    results_is["Grid Multi"] = GridBacktestEngine(df_train, capital=args.capital, commission=args.commission,
        slippage=args.slippage, grid_pct=args.grid_pct, n_levels=args.grid_levels,
        interval_factor=interval_factor, min_notional=args.min_notional,
        stop_loss_pct=args.stop_loss).run()
    results_oos["Grid Multi"] = GridBacktestEngine(df_test, capital=args.capital, commission=args.commission,
        slippage=args.slippage, grid_pct=args.grid_pct, n_levels=args.grid_levels,
        interval_factor=interval_factor, min_notional=args.min_notional,
        stop_loss_pct=args.stop_loss).run()
    print(f"  {GREEN}✔{RESET}  Grid Multi")

    print_results_cmp(results_is, results_oos, args.capital)

    if args.log_file:
        for name in results_is:
            is_r = results_is[name]
            oos_r = results_oos[name]
            logging.info("%-20s | IS: ret=%+.1f%% sharpe=%.2f dd=%.1f%% | OOS: ret=%+.1f%% sharpe=%.2f dd=%.1f%%",
                         name, is_r["total_return"], is_r["sharpe"], is_r["max_drawdown"],
                         oos_r["total_return"], oos_r["sharpe"], oos_r["max_drawdown"])

    # 7. Mostrar trades del mejor (por Sharpe OOS)
    best_name = max(results_oos, key=lambda k: results_oos[k]["sharpe"])
    print_trade_summary(best_name, results_oos[best_name]["trades"])

    # 8. Optimización opcional
    if args.optimize:
        strat = args.optimize_strategy
        optimizer = OPTIMIZERS.get(strat)
        if optimizer is None:
            print(f"{RED}Estrategia '{strat}' no válida. Opciones: {list(OPTIMIZERS.keys())}{RESET}")
        else:
            extra_kwargs = {}
            if strat == "grid_multi":
                extra_kwargs["grid_levels"] = args.grid_levels
            best_params = optimizer(df_train, args.capital, interval_factor,
                                    slippage=args.slippage, stop_loss_pct=args.stop_loss,
                                    allow_short=args.allow_short, vol_factor=args.vol_filter,
                                    min_notional=args.min_notional,
                                    **extra_kwargs)

            # Validación OOS solo para la estrategia optimizada
            oos_config = {
                "rsi":       (strategy_rsi,            lambda p: {"oversold": p["oversold"], "overbought": p["overbought"]}),
                "ema":       (strategy_ema_cross,      lambda p: {"fast": p["fast"], "slow": p["slow"]}),
                "bollinger": (strategy_bollinger,      lambda p: {"std": p["std"]}),
                "grid":      (strategy_grid,           lambda p: {"grid_pct": p["grid_pct"]}),
            }

            print(f"\n{CYAN}🧪 Validando parámetros optimizados en out-of-sample (30%)...{RESET}")

            if strat == "grid_multi":
                best_pct = best_params.get("grid_pct", 0.01)
                grid_engine = GridBacktestEngine(df_test, capital=args.capital, commission=args.commission,
                    slippage=args.slippage, grid_pct=best_pct, n_levels=args.grid_levels,
                    interval_factor=interval_factor, min_notional=args.min_notional,
                    stop_loss_pct=args.stop_loss)
                res_oos = grid_engine.run()
            else:
                engine_oos = BacktestEngine(df_test, **engine_kwargs)
                sig_fn, sig_kwargs_fn = oos_config[strat]
                sig_kwargs = sig_kwargs_fn(best_params)
                sig_oos = sig_fn(df_test, **sig_kwargs)
                if args.vol_filter > 0:
                    sig_oos = filter_signals(df_test, sig_oos, args.vol_filter)
                res_oos = engine_oos.run(sig_oos)

            print(f"\n  {'In-sample':20s}  Sharpe={best_params['sharpe']:.2f}  Return={best_params['total_return']:+.1f}%")
            color = GREEN if res_oos["total_return"] >= 0 else RED
            print(f"  {'Out-of-sample':20s}  Sharpe={res_oos['sharpe']:.2f}  {color}Return={res_oos['total_return']:+.1f}%{RESET}")

            if res_oos["sharpe"] > 0.8 * best_params["sharpe"]:
                print(f"\n  {GREEN}✔ La estrategia mantiene rendimiento en datos no vistos. Buena señal.{RESET}")
            else:
                print(f"\n  {YELLOW}⚠  Degradación significativa en OOS. Posible overfitting. Seguir iterando.{RESET}")

            if args.log_file:
                logging.info("Optimize %s: params=%s", strat, {k: v for k, v in best_params.items() if k not in ("equity", "drawdowns", "trades")})
                logging.info("Optimize IS:  ret=%+.1f%% sharpe=%.2f", best_params["total_return"], best_params["sharpe"])
                logging.info("Optimize OOS: ret=%+.1f%% sharpe=%.2f", res_oos["total_return"], res_oos["sharpe"])

    # 9. Exportar
    if args.export:
        export_csv({"IS": results_is, "OOS": results_oos}, args.pair, trade_log=args.trade_log)

    print(f"\n{BOLD}{'═'*80}")
    print(f"  PRÓXIMOS PASOS RECOMENDADOS")
    print(f"{'═'*80}{RESET}")
    print(f"""
  1. Probá timeframes cortos (5m-15m) con --interval para decenas de trades/día
  2. Ajustá --stop-loss y --slippage para simular condiciones reales
  3. Corré con --allow-short para estrategias long/short
  4. Optimizá la mejor estrategia con --optimize --optimize-strategy=<nombre>
  5. Si Sharpe > 1.0 y OOS estable → pasá a paper trading en Binance Testnet
""")


if __name__ == "__main__":
    main()
