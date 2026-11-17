# Deploying AdLens

Three ways to put the app online, from easiest to most flexible. All of them
use the same code; behaviour is set with environment variables (below).

| Route | Cost | HTTPS link | Speed | Accounts survive restarts |
|---|---|---|---|---|
| A. Hugging Face Spaces (recommended) | Free CPU tier | Automatic | Analysis fast; rewrites a few seconds; Stable Diffusion off | Only with persistent storage (paid) or `SESSION_SECRET` + re-sign-up |
| B. Cloud server (Docker + Caddy) | Pay per hour / student credits | Caddy, automatic | Depends on the machine; GPU makes everything fast | Yes (Docker volume) |
| C. Your Mac + Cloudflare Tunnel | Free | Automatic | Fast (uses the Mac's GPU) | Yes, but only while the Mac is on |

Free tiers and prices change - check the provider's current page before you choose.

## Settings (environment variables)

| Variable | Default | What it does |
|---|---|---|
| `SIGNUP_MODE` | `open` (`invite` in the Docker image) | `open` = anyone can sign up, `invite` = needs `INVITE_CODE`, `closed` = no new accounts |
| `INVITE_CODE` | empty | The code people must enter to sign up when `SIGNUP_MODE=invite` (keep it secret) |
| `PAGESPEED_API_KEY` | empty | Google PageSpeed key for Website Audit speed scores (keep it secret) |
| `SESSION_SECRET` | random file | Any long random text. Keeps people logged in across restarts (keep it secret) |
| `LITE_MODE` | `0` | `1` = no Stable Diffusion and the small flan-t5 rewriter: about 3-4 GB RAM instead of 8-16 |
| `DISABLE_GENERATIVE` | `0` (`1` in the Docker image) | `1` = Stable Diffusion off (it takes minutes per image on a CPU) |
| `RATE_LIMIT_PER_MINUTE` | `30` | Analyses each user may run per minute (`0` = no limit) |
| `AUTH_LIMIT_PER_MINUTE` | `10` | Log-in / sign-up attempts per IP address per minute |
| `MAX_CONCURRENT_GENERATIONS` | `1` | Stable Diffusion images generated at once, across all users |
| `MAX_CONCURRENT_BULK_JOBS` | `2` | Bulk CSV jobs running at once, across all users |
| `BULK_MAX_ROWS` | `20000` | Largest CSV accepted |
| `COOKIE_SECURE` | `0` | `1` = log-in cookie only over HTTPS. Turned on automatically behind an HTTPS proxy |
| `ACCOUNTS_DIR` | `data/accounts` | Where accounts are stored, e.g. `/data/accounts` on persistent storage |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Where the server listens (`0.0.0.0` / `7860` in the Docker image) |

On a laptop these go in `.env`; on a host, in its settings or secrets page. `.env`
is git-ignored and in `.dockerignore`, so secrets never reach GitHub or the image.

## A. Hugging Face Spaces

1. Create a free account at https://huggingface.co.
2. **New Space** -> name it (e.g. `adlens`) -> SDK **Docker** -> template **Blank** ->
   hardware **CPU basic** (free) -> visibility Public or Private -> **Create Space**.
3. In the Space: **Settings -> Variables and secrets**, add as **Secrets**:
   - `INVITE_CODE` - the code you'll give your panel and friends
   - `PAGESPEED_API_KEY` - your Google key
   - `SESSION_SECRET` - any long random text (e.g. 40 random letters)
4. **Settings -> Access Tokens** on your profile -> create a token with **write** access.
5. From the project folder on your Mac (after committing your changes):
   ```bash
   ./deploy/push_to_hf_space.sh <your-hf-username>/adlens
   ```
   Git asks for a username (your Hugging Face name) and password (the token).
6. The Space builds the image and downloads the models - expect 20-40 minutes the
   first time. When it says **Running**, the app is at
   `https://<your-hf-username>-adlens.hf.space`.
7. Sign up there with your invite code.

Good to know:
- A free Space **sleeps after a period without visitors** and wakes on the next
  visit (the first page then takes a minute or two).
- A free Space's disk is **wiped on every restart**, so accounts and history are
  lost; people sign up again with the invite code. Persistent storage (paid) +
  `ACCOUNTS_DIR=/data/accounts` keeps them.
- **GPU:** switch the Space's hardware to a GPU (paid, per hour), change
  `TORCH_INDEX` in the Dockerfile to `https://download.pytorch.org/whl/cu121`, and set
  the Variable `DISABLE_GENERATIVE=0` to turn Stable Diffusion back on. Push again.
- **Less memory:** set the Variable `LITE_MODE=1`.
- To update the live site later: commit, then run step 5 again.

## B. Cloud server (Ubuntu VM + Docker + Caddy)

Use a VM with at least 16 GB RAM and 40 GB disk (or 8 GB RAM with `LITE_MODE=1`).
Student credits: GitHub Student Developer Pack, Azure for Students.

1. Install Docker: https://docs.docker.com/engine/install/ubuntu/
2. Get the code and create the settings file:
   ```bash
   git clone https://github.com/kashvi1504/AI-Driven-Multimodal-Marketing-Assistant.git
   cd AI-Driven-Multimodal-Marketing-Assistant
   nano .env     # SIGNUP_MODE=invite, INVITE_CODE=..., PAGESPEED_API_KEY=..., SESSION_SECRET=...
   ```
3. Build and run (the app listens only on the server itself; Caddy faces the internet):
   ```bash
   docker build -t adlens .
   docker run -d --name adlens --restart unless-stopped \
     -p 127.0.0.1:8000:7860 --env-file .env \
     -e ACCOUNTS_DIR=/home/user/app/data/accounts -v ss-data:/home/user/app/data \
     adlens
   ```
   With an NVIDIA GPU: build with `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121`,
   run with `--gpus all`, and add `DISABLE_GENERATIVE=0` to `.env`.
4. HTTPS: point your domain's DNS A record at the server, install Caddy
   (https://caddyserver.com/docs/install), edit the domain in `deploy/Caddyfile`, then:
   ```bash
   sudo cp deploy/Caddyfile /etc/caddy/Caddyfile && sudo systemctl reload caddy
   ```
   Caddy fetches and renews the certificate itself.
5. Firewall: allow only ports 22, 80 and 443 (e.g. `sudo ufw allow OpenSSH && sudo ufw allow 80,443/tcp && sudo ufw enable`).

## C. Your Mac + Cloudflare Tunnel (quick public demo)

1. `brew install cloudflared`
2. Start the app as usual (`.venv/bin/python 03_web_app.py`), with `SIGNUP_MODE=invite`
   and `INVITE_CODE=...` in `.env`.
3. In a second Terminal: `cloudflared tunnel --url http://localhost:8000`
4. It prints a `https://....trycloudflare.com` link - share that. It works while both
   commands keep running; the link changes each time.

## Security checklist for a public site

- [ ] `SIGNUP_MODE=invite` with a strong `INVITE_CODE` (or `closed`)
- [ ] Served over HTTPS (A and C do this; B uses Caddy)
- [ ] `PAGESPEED_API_KEY`, `INVITE_CODE`, `SESSION_SECRET` set as secrets, never committed
- [ ] Rate limits left on (`RATE_LIMIT_PER_MINUTE`, `AUTH_LIMIT_PER_MINUTE`)
- [ ] Stable Diffusion off unless the host has a GPU (`DISABLE_GENERATIVE=1`)
- [ ] The PageSpeed key restricted to the PageSpeed Insights API in Google Cloud

Limits that remain: accounts are stored in a JSON file and limits are kept in memory,
which suits one server process and a class-sized audience, not thousands of users.
