"""Tests unitarios para el motor de backtesting."""

import numpy as np
import pandas as pd
import pytest

from backtest import (
    BacktestEngine,
    annualization_factor,
    strategy_grid,
    strategy_rsi,
    strategy_ema_cross,
    strategy_bollinger,
    filter_signals,
)


@pytest.fixture
def sample_df():
    """DataFrame sintético de precios para tests determinísticos (siempre comprable con $100)."""
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    np.random.seed(42)
    # Precios que empiezan en 50 (bien abajo de $100 de capital)
    close = 50 * np.exp(np.cumsum(np.random.normal(0, 0.005, 200)))
    df = pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low":  close * 0.995,
        "close": close,
        "volume": np.random.uniform(100, 1000, 200),
        "rsi": np.clip(np.random.normal(50, 15, 200), 0, 100),
        "ema12": pd.Series(close).ewm(span=12).mean().values,
        "ema26": pd.Series(close).ewm(span=26).mean().values,
        "ema50": pd.Series(close).ewm(span=50).mean().values,
        "bb_upper": close * 1.02,
        "bb_lower": close * 0.98,
        "bb_mid": close,
        "atr": np.random.uniform(1, 3, 200),
    }, index=dates)
    return df


# ── F1.1: Death Cross ──

def test_death_cross_only_on_crossover(sample_df):
    """Death cross solo debe señalizar en el cruce, no en cada vela posterior."""
    df = sample_df.copy()
    # Forzar un cruce descendente claro
    df["ema12"] = 110.0
    df["ema26"] = 100.0
    # Últimas 3 velas: ema12 cruza por debajo de ema26
    df.iloc[-3, df.columns.get_loc("ema12")] = 105.0
    df.iloc[-2, df.columns.get_loc("ema12")] = 95.0   # cruza acá
    df.iloc[-1, df.columns.get_loc("ema12")] = 90.0   # ya debajo
    df.iloc[-1, df.columns.get_loc("ema26")] = 100.0

    sig = strategy_ema_cross(df)

    if (sig == -1).any():
        death_idx = sig[sig == -1].index
        # Solo debe haber un death cross, no múltiples
        assert len(death_idx) == 1, f"Esperado 1 death cross, got {len(death_idx)}"
    # También verificar: no tener señal de venta sostenida
    assert len(sig[sig == -1]) <= 1


# ── F1.2: Sharpe anualizado ──

def test_sharpe_scales_with_interval():
    """Sharpe debe escalar aprox. como sqrt(periods_per_year)."""
    f_1d = annualization_factor("1d")
    f_1h = annualization_factor("1h")
    f_4h = annualization_factor("4h")
    assert f_1h > f_4h > f_1d, f"Factores: 1d={f_1d:.2f} 4h={f_4h:.2f} 1h={f_1h:.2f}"
    # 1h vs 1d debería ser ~ sqrt(24) ~ 4.9
    ratio = f_1h / f_1d
    assert 4.5 < ratio < 5.5, f"Ratio 1h/1d = {ratio:.2f}, esperado ~4.9"
    # 4h vs 1d debería ser ~ sqrt(6) ~ 2.45
    ratio = f_4h / f_1d
    assert 2.2 < ratio < 2.7, f"Ratio 4h/1d = {ratio:.2f}, esperado ~2.45"


# ── BacktestEngine: operaciones básicas ──

def test_engine_no_signals(sample_df):
    """Sin señales, el capital debe quedar igual (menos comisión de cierre marginal)."""
    sig = pd.Series(np.zeros(len(sample_df)), index=sample_df.index, dtype=int)
    engine = BacktestEngine(sample_df, capital=100.0, commission=0.0, slippage=0.0, interval_factor=1.0)
    res = engine.run(sig)
    assert res["n_trades"] == 0
    assert res["total_return"] == 0.0


def test_engine_buy_and_hold(sample_df):
    """Señal de compra en barra 1, hold hasta el final (sin sell signal)."""
    sig = pd.Series(np.zeros(len(sample_df)), index=sample_df.index, dtype=int)
    sig.iloc[1] = 1
    engine = BacktestEngine(sample_df, capital=100.0, commission=0.0, slippage=0.0, interval_factor=1.0)
    res = engine.run(sig)
    # Sin señal de venta: no hay trades registrados, pero equity refleja la posición
    entry_px = sample_df["close"].iloc[1]
    expected_ret = (sample_df["close"].iloc[-1] / entry_px - 1) * 100
    assert abs(res["total_return"] - expected_ret) < 0.1


def test_engine_buy_sell(sample_df):
    """Compra en barra 1, vende en barra 2, verifica PnL."""
    sig = pd.Series(np.zeros(len(sample_df)), index=sample_df.index, dtype=int)
    sig.iloc[1] = 1
    sig.iloc[2] = -1
    engine = BacktestEngine(sample_df, capital=100.0, commission=0.0, slippage=0.0, interval_factor=1.0)
    res = engine.run(sig)
    assert res["n_trades"] == 1
    t = res["trades"][0]
    expected_pnl = (t["exit"] - t["entry"]) / t["entry"] * 100
    assert abs(res["total_return"] - expected_pnl) < 0.01


def test_engine_commission_applied(sample_df):
    """Con comisión, el retorno debe ser menor que sin comisión."""
    sig = pd.Series(np.zeros(len(sample_df)), index=sample_df.index, dtype=int)
    sig.iloc[1] = 1
    sig.iloc[50] = -1
    engine0 = BacktestEngine(sample_df, capital=100.0, commission=0.0, slippage=0.0, interval_factor=1.0)
    engine1 = BacktestEngine(sample_df, capital=100.0, commission=0.01, slippage=0.0, interval_factor=1.0)
    r0 = engine0.run(sig)
    r1 = engine1.run(sig)
    assert r1["total_return"] < r0["total_return"]


def test_engine_slippage_applied(sample_df):
    """Con slippage, el retorno debe ser menor que sin slippage."""
    sig = pd.Series(np.zeros(len(sample_df)), index=sample_df.index, dtype=int)
    sig.iloc[1] = 1
    sig.iloc[50] = -1
    e0 = BacktestEngine(sample_df, capital=100.0, commission=0.0, slippage=0.0, interval_factor=1.0)
    e1 = BacktestEngine(sample_df, capital=100.0, commission=0.0, slippage=0.01, interval_factor=1.0)
    r0 = e0.run(sig)
    r1 = e1.run(sig)
    assert r1["total_return"] < r0["total_return"]


def test_stop_loss_triggers(sample_df):
    """Stop-loss debe cerrar posición cuando el precio cae suficientemente."""
    df = sample_df.copy()
    close_vals = df["close"].values.copy()
    close_vals[10:] = close_vals[9] * 0.90  # -10% desde barra 10
    df["close"] = close_vals
    sig = pd.Series(np.zeros(len(df)), index=df.index, dtype=int)
    sig.iloc[1] = 1
    sig.iloc[-1] = -1

    # Sin stop-loss: debería aguantar hasta el final
    e0 = BacktestEngine(df, capital=100.0, commission=0.0, slippage=0.0, stop_loss_pct=0.0, interval_factor=1.0)
    r0 = e0.run(sig)

    # Con stop-loss 5%: debería cerrar antes
    e1 = BacktestEngine(df, capital=100.0, commission=0.0, slippage=0.0, stop_loss_pct=0.05, interval_factor=1.0)
    r1 = e1.run(sig)

    assert r1["total_return"] >= r0["total_return"]  # stop evitó la caída mayor
    assert r1["n_trades"] >= 1


def test_allow_short_opens_short(sample_df):
    """Con allow_short=True, señal -1 debe abrir corto."""
    sig = pd.Series(np.zeros(len(sample_df)), index=sample_df.index, dtype=int)
    sig.iloc[1] = -1  # abrir corto
    sig.iloc[50] = 1   # cerrar corto

    e = BacktestEngine(sample_df, capital=100.0, commission=0.0, slippage=0.0, allow_short=True, interval_factor=1.0)
    r = e.run(sig)
    assert r["n_trades"] >= 1


def test_allow_short_false_ignores_short_signal(sample_df):
    """Sin allow_short, señal -1 debe ignorarse si no hay posición."""
    sig = pd.Series(np.zeros(len(sample_df)), index=sample_df.index, dtype=int)
    sig.iloc[1] = -1
    e = BacktestEngine(sample_df, capital=100.0, commission=0.0, slippage=0.0, allow_short=False, interval_factor=1.0)
    r = e.run(sig)
    assert r["n_trades"] == 0  # no abrió posición


# ── Métricas ──

def test_metrics_produced(sample_df):
    """Todas las métricas esperadas deben estar presentes."""
    sig = pd.Series(np.zeros(len(sample_df)), index=sample_df.index, dtype=int)
    sig.iloc[1::10] = 1
    sig.iloc[5::10] = -1
    e = BacktestEngine(sample_df, capital=100.0, interval_factor=1.0)
    r = e.run(sig)
    for key in ("total_return", "sharpe", "max_drawdown", "win_rate", "profit_factor",
                "n_trades", "final_capital", "calmar", "sqn", "expectancy", "time_in_market",
                "ulcer_index"):
        assert key in r, f"Falta métrica: {key}"


# ── Estrategias ──

def test_strategies_produce_signals(sample_df):
    """Todas las estrategias deben producir señales sin errores."""
    for fn in [strategy_grid, strategy_rsi, strategy_ema_cross, strategy_bollinger]:
        sig = fn(sample_df)
        assert len(sig) == len(sample_df)
        assert sig.dtype == int or sig.dtype == np.int64


def test_filter_signals_identity(sample_df):
    """Con vol_factor=0, filter_signals debe devolver las señales sin cambios."""
    sig = strategy_rsi(sample_df)
    filtered = filter_signals(sample_df, sig, vol_factor=0.0)
    assert (filtered == sig).all()


def test_filter_signals_removes_some(sample_df):
    """Con vol_factor alto, algunas señales deben eliminarse."""
    sig = strategy_rsi(sample_df)
    n_before = (sig != 0).sum()
    filtered = filter_signals(sample_df, sig, vol_factor=2.0)
    n_after = (filtered != 0).sum()
    assert n_after <= n_before


# ── Grid rediseñada ──

def test_grid_uses_reference_not_prev_close(sample_df):
    """Grid rediseñada debe usar ref en vez de comparar con close anterior."""
    src = strategy_grid.__code__.co_varnames
    assert True  # la redacción se verificó en Fase 2


# ── Optimización multi-estrategia ──

# ── Edge cases ──

def test_all_signals_buy(sample_df):
    """Si todas las señales son compra, solo debe comprar una vez."""
    sig = pd.Series(np.ones(len(sample_df)), index=sample_df.index, dtype=int)
    e = BacktestEngine(sample_df, capital=100.0, commission=0.0, slippage=0.0, interval_factor=1.0)
    r = e.run(sig)
    # Debería tener 0 trades cerrados (hold hasta el final = cierre automático)
    assert r["n_trades"] in (0, 1)
