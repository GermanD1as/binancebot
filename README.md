# Crypto Backtest Framework

Motor de backtesting para estrategias de trading crypto usando datos reales de Binance.

## Instalación

```bash
pip install ccxt pandas pandas-ta tabulate colorama
```

## Uso básico

```bash
# Configuración por defecto (BTC/USDT, 1 año, $500)
python backtest.py

# Cambiar par
python backtest.py --pair ETHUSDT
python backtest.py --pair SOLUSDT

# Cambiar intervalo de velas
python backtest.py --interval 4h
python backtest.py --interval 1h

# Más historia
python backtest.py --days 730

# Con optimización de parámetros (in-sample / out-of-sample)
python backtest.py --optimize

# Exportar equity curves a CSV
python backtest.py --export

# Todo junto
python backtest.py --pair ETHUSDT --interval 4h --days 730 --capital 300 --optimize --export
```

## Estrategias incluidas

| Estrategia | Lógica |
|---|---|
| Grid Trading | Compra en caídas de X%, vende en rebotes de X% |
| RSI Mean Reversion | Compra RSI < 32, vende RSI > 68 |
| EMA Crossover | Compra golden cross EMA 12/26, vende death cross |
| Bollinger Bands | Compra en banda inferior, vende en banda superior |

## Métricas clave

- **Sharpe Ratio**: retorno ajustado por riesgo. > 1.0 es bueno
- **Max Drawdown**: peor caída desde pico. Idealmente < 20%
- **Profit Factor**: ganancia total / pérdida total. > 1.5 es sólido
- **Win Rate**: % de trades ganadores (no tan importante solo)

## Hoja de ruta

```
Paso 1: backtest.py → encontrar estrategia ganadora
Paso 2: backtest.py --optimize → validar sin overfitting
Paso 3: Binance Testnet (paper trading, sin dinero real)
Paso 4: Live con $50-100, monitoreo activo
Paso 5: Escalar si rentable 2 meses seguidos
```
