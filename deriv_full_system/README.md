# Deriv AI Signal System

> Real-time AI-powered trading signal generator for Deriv synthetic indices.
> **Signals only — no auto-execution. You stay in full control.**

---

## What it does

- Connects to Deriv via WebSocket and streams live tick data
- Builds 1-minute OHLC candles in real time
- Computes RSI, MACD, EMA, Bollinger Bands, ATR, momentum, velocity
- Runs a logistic regression AI model scoring price direction probability
- Emits BUY/SELL signals with confidence score, entry, stop-loss, take-profit
- Sends alerts to your Telegram bot instantly
- Filters out low-quality signals (flat markets, choppy conditions, low confidence)
- Learns from outcomes — model improves as it runs

---

## Project structure

```
deriv_signal_system/
├── main.py              ← Entry point (run this)
├── config.py            ← All settings via environment variables
├── data_collector.py    ← Deriv WebSocket + candle builder
├── model.py             ← Feature engineering + AI model
├── signal_engine.py     ← Signal generation + filtering + risk
├── risk_manager.py      ← ATR-based SL/TP + position sizing
├── notifier.py          ← Telegram bot + console fallback
├── requirements.txt     ← Python dependencies
├── Procfile             ← Railway process definition
├── railway.json         ← Railway deployment config
├── nixpacks.toml        ← Python 3.11 build config
├── runtime.txt          ← Python version pin
├── .env.example         ← Template for your secrets
├── .gitignore           ← Keeps secrets out of git
└── README.md
```

---

## Quick start (local)

```bash
# 1. Clone your repo
git clone https://github.com/YOUR_USERNAME/deriv-signal-system.git
cd deriv-signal-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env and fill in DERIV_API_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# 4. Run
python main.py
```

---

## Deploy to Railway (step by step)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit — Deriv AI Signal System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/deriv-signal-system.git
git push -u origin main
```

### Step 2 — Create Railway project

1. Go to [railway.app](https://railway.app) and sign in with GitHub
2. Click **New Project** → **Deploy from GitHub repo**
3. Select your `deriv-signal-system` repository
4. Railway detects `Procfile` automatically — click **Deploy**

### Step 3 — Set environment variables

In your Railway project dashboard → **Variables** tab, add:

| Variable | Value |
|----------|-------|
| `DERIV_API_TOKEN` | Your Deriv API token |
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID |
| `SYMBOLS` | `R_75,R_100,1HZ100V,BOOM1000,CRASH1000` |
| `CONFIDENCE_THRESHOLD` | `0.68` |
| `COOLDOWN_SECONDS` | `180` |
| `RISK_PCT` | `1.0` |
| `RR_RATIO` | `2.0` |

Railway will automatically redeploy when you save variables.

### Step 4 — Verify

- Go to **Deployments** tab → click your deployment → **View logs**
- You should see: `Deriv AI Signal System — starting`
- Then: `Authorised as [your login]`
- Then: `Subscribed to R_75` (and each symbol)
- Within a few minutes: signals appear in your Telegram

---

## Getting your API keys

### Deriv API token

1. Log into [app.deriv.com](https://app.deriv.com)
2. Profile → **Settings** → **Security & Safety** → **API token**
3. Click **Create new token**
4. Enable scopes: ✅ **Read** + ✅ **Trading information**
5. Copy the token → paste as `DERIV_API_TOKEN`

### Telegram bot token + chat ID

1. Open Telegram → search **@BotFather** → send `/newbot`
2. Follow prompts → copy the bot token → paste as `TELEGRAM_BOT_TOKEN`
3. Start a chat with your new bot (search its username, send any message)
4. Open Telegram → search **@userinfobot** → send `/start`
5. It replies with your chat ID → paste as `TELEGRAM_CHAT_ID`

---

## Telegram signal format

```
📈 🟢 BUY  |  Volatility 75 Index

🎯 Confidence:  ▓▓▓▓▓▓▓░░░  72%
🤖 Model: HEURISTIC

🔖 Entry:        12453.84200
🛑 Stop loss:    12437.21100
💰 Take profit:  12487.08400
⚖️  Risk:Reward:  1 : 2.0

📊 Indicators: RSI 31.4 | MACD +0.00312 | EMA9/21 ↑

💡 Reason: RSI oversold (31) + EMA9 > EMA21 ↑ + MACD bullish
⏱ Timeframe: 1m

⚠️ This is a signal, not financial advice. Always manage your own risk.
```

## Bot commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/status` | Uptime, total signals, last signal per symbol |
| `/signals` | Last 5 signals sent |
| `/help` | Command list |

---

## How the AI model works

1. **Features (12 inputs):** RSI normalised, MACD sign + histogram, EMA 9/21 cross + slope, Bollinger Band position + width, ATR, momentum, velocity, trend strength, volatility spike ratio
2. **Cold start:** Hand-tuned logistic regression weights work immediately from minute one — no training data required
3. **Online learning:** 60 seconds after each signal the system checks actual price movement and updates the model weights via SGD partial fit
4. **Persistence:** Model saved to disk every 5 minutes and reloaded on restart

## Signal filters

| Filter | Default | What it blocks |
|--------|---------|----------------|
| Confidence threshold | 68% | Weak/uncertain model predictions |
| Cooldown | 180 s | Overtrading same symbol |
| BB width | < 0.08% | Flat, low-volatility markets |
| RSI neutral + weak trend | RSI 45–55 | Choppy, directionless markets |
| RR ratio | < 1:2 | Poor reward vs risk setups |

---

## Disclaimer

> Trading signals are generated by a statistical model and **do not guarantee profits**.
> Synthetic indices are high-risk instruments. Past signal accuracy does not predict future results.
> Always apply your own risk management. Never risk money you cannot afford to lose.
