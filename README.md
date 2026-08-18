# Kronos XAU/USD Forecast API (Render)

A small FastAPI service that wraps the [Kronos](https://github.com/shiyu-coder/Kronos) foundation
model to forecast future XAU/USD candles from historical OHLCV data. This is the inference
backend for the standalone gold signal dashboard — it does not know about your account balance,
risk settings, or place any trades. It just turns candles into forecasted candles.

Runs on **Render's free web service tier** — genuinely free, no card required. The trade-off:
free tier gives 512MB RAM, which is tight for PyTorch, so this is configured to use
**`kronos-mini`** (4.1M params) rather than the larger `kronos-small`/`kronos-base` — smaller
model, noticeably faster, lower forecast quality than the bigger variants but comfortably fits
in memory.

## Deploy (~10 minutes, no card, no local installs)

1. Push this `render-kronos-api` folder to a GitHub repo (create a new repo, e.g.
   `kronos-xau-api`, and push the contents of this folder to it — Render deploys from Git).
2. Go to https://render.com and sign up free (GitHub sign-in is easiest — no card needed).
3. Click **New +** → **Web Service**, connect your GitHub account, and pick the repo you just
   pushed.
4. Render will detect the `Dockerfile` automatically. Set:
   - **Instance type:** Free
   - **Environment variables** (in the Render dashboard, add these two):
     - `API_KEY` = a random string you make up (locks down `/forecast` so strangers can't hit it)
     - `KRONOS_MODEL` = `kronos-mini` (already the default, but explicit is safer)
5. Click **Create Web Service**. First build takes 5-10 minutes (installing PyTorch CPU +
   downloading model weights).
6. Once it shows **Live**, Render gives you a URL like `https://kronos-xau-api.onrender.com`.
   Copy this — it's what goes into `KRONOS_API_URL` on the Netlify side.
7. Test it:
   ```bash
   curl https://kronos-xau-api.onrender.com/health
   ```
   You should see `{"status":"ok", ...}`.

## Notes on the free tier

- Free Render services **spin down after 15 minutes of inactivity** and take 30-60s to wake back
  up on the next request. The Netlify function already has a generous timeout and a clear
  "waking up" error message baked in for this.
- 512MB RAM is genuinely tight. `kronos-mini` fits; if you ever see the service crash/restart
  right after a request (check the Render **Logs** tab for an "out of memory" or killed-process
  message), that's this limit being hit — the fix is upgrading to Render's Starter plan ($7/mo,
  1GB+ RAM) and switching `KRONOS_MODEL` to `kronos-small` for better quality.
- To redeploy after any code change, just push to your GitHub repo — Render auto-deploys on push.
