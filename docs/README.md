# Crypto Backtest Framework — Bot Testnet

Framework de backtesting y trading automatizado para Binance (15m, multi-par, multi-estrategia con filtro de tendencia 1h).

## Setup

```bash
pip install ccxt pandas numpy ta tabulate colorama pytest
```

## Backtest

```bash
# Portafolio walk-forward con 3 estrategias (Grid + RSI + Regime) en BNB
python backtest.py --multi-pair BNB/USDT --interval 15m --days 365 --capital 1000 --stop-loss 0.02 --allow-short --grid-pct 0.01 --grid-levels 3

# Multi-par
python backtest.py --multi-pair BTC/USDT,BNB/USDT,SOL/USDT --interval 15m --days 365 --capital 1000 --stop-loss 0.02 --allow-short
```

### Barrido paramétrico

```bash
python backtest.py --pair BNB/USDT --interval 15m --days 365 --sweep grid
python backtest.py --pair BNB/USDT --interval 15m --days 365 --allow-short --sweep rsi
```

### Tests

```bash
pytest test_backtest.py -q
```

## Testnet

Requiere API keys de [Binance Testnet](https://testnet.binance.com/).

```bash
export BINANCE_TESTNET_API_KEY="tu_key"
export BINANCE_TESTNET_API_SECRET="tu_secret"

# Grid:BNB (recomendada: 6/9 ventanas walk-forward, OOS Sharpe 1.40)
python live.py --strategy grid

# RSI:BNB (5/9 ventanas, OOS Sharpe 0.72)
python live.py --strategy rsi

# Sin ejecutar órdenes reales
python live.py --strategy grid --dry-run
```

### Automatización (cron)

Ejecuta cada 15 minutos (sincronizado con velas de Binance):

```
# crontab -e
*/15 * * * * cd /ruta/del/proyecto && python live.py --strategy grid >> logs/grid.log 2>&1
*/15 * * * * cd /ruta/del/proyecto && python live.py --strategy rsi >> logs/rsi.log 2>&1
```

### Verificación

```bash
# Estado del bot (posiciones, PnL, trades)
python3 -c "
import pickle, os
p='cache/live_grid_BNB_USDT.pkl'
if os.path.exists(p):
    st=pickle.load(open(p,'rb'))
    t=st['pnl_history']
    w=sum(1 for x in t if x>0)
    print(f'Trades: {len(t)} | Win: {w/len(t)*100:.0f}% | PnL: \${sum(t):.2f}' if t else 'Sin trades aun')
    print(f'Cash: \${st[\"cash\"]:.2f} | Posiciones: {len(st[\"positions\"])}')
"

# Últimas líneas del log
tail -5 logs/grid_$(date +%Y%m%d).log

# Cache de datos
ls -lh cache/*.pkl | grep -v live
```

## Validación multi-par (walk-forward 9 ventanas, 365d)

| Par | Grid wins | RSI wins | Mejor estrategia |
|---|---|---|---|
| **BNB/USDT** | **6/9** | **5/9** | Grid |
| BTC/USDT | 0/9 | 2/9 | Ninguna |
| ETH/USDT | 1/9 | 2/9 | Ninguna |
| SOL/USDT | 1/9 | 4/9 | RSI (marginal) |

**Conclusión:** Grid y RSI solo son consistentes en BNB. Las estrategias están calibradas para su dinámica específica. No operar en otros pares sin recalibrar.

## Estrategias validadas (walk-forward 9 ventanas, BNB 365d)

| Estrategia | Wins | IS:Sh | OOS:Sh | OOS:PF | Alloc |
|---|---|---|---|---|---|
| Grid con MTF | 6/9 | 0.71 | 1.40 | 1.51 | ~60% |
| RSI con MTF | 5/9 | 0.15 | 0.72 | 1.53 | ~40% |

Grid:BNB con filtro de tendencia 1h es la estrategia más robusta. Valida en 6 de 9 ventanas walk-forward con Sharpe OOS de 1.40.

## Estructura

```
├── backtest.py       # Framework: engines, PM, sweeps, CLI
├── live.py           # Bot testnet (Grid/RSI en BNB)
├── test_backtest.py  # Tests unitarios
├── cache/            # OHLCV + estado del bot
└── docs/README.md    # Este archivo
```
