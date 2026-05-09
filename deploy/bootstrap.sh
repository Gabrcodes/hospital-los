#!/usr/bin/env bash
# First-run setup on a fresh Ubuntu 24.04 EC2 instance.
# Idempotent — safe to re-run.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Gabrcodes/hospital-los.git}"
APP_DIR="/opt/hospital-los"
DOMAIN="${DOMAIN:-dsc.gabr.online}"

echo "==> Updating apt and installing system packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3 python3-venv python3-pip git curl debian-keyring debian-archive-keyring apt-transport-https

echo "==> Installing Caddy (reverse proxy + auto-HTTPS)"
if ! command -v caddy >/dev/null 2>&1; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | \
    sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | \
    sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq caddy
fi

echo "==> Cloning / updating repository at $APP_DIR"
if [ ! -d "$APP_DIR/.git" ]; then
  sudo git clone "$REPO_URL" "$APP_DIR"
else
  sudo git -C "$APP_DIR" pull --ff-only
fi
sudo chown -R "$USER":"$USER" "$APP_DIR"

echo "==> Creating Python virtualenv and installing dependencies"
cd "$APP_DIR"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

echo "==> Training models (one-shot, generates saved_models/*.pkl and plots/*.png)"
if [ ! -f saved_models/regression_model.pkl ]; then
  .venv/bin/python models.py
fi

echo "==> Installing systemd unit for Streamlit"
sudo cp deploy/streamlit.service /etc/systemd/system/streamlit.service
sudo systemctl daemon-reload
sudo systemctl enable --now streamlit.service

echo "==> Configuring Caddy reverse proxy for $DOMAIN"
sudo install -m 644 deploy/Caddyfile /etc/caddy/Caddyfile
sudo sed -i "s|__DOMAIN__|$DOMAIN|g" /etc/caddy/Caddyfile
sudo systemctl reload caddy || sudo systemctl restart caddy

echo
echo "==> Bootstrap complete."
echo "    Streamlit unit:  $(systemctl is-active streamlit.service)"
echo "    Caddy unit:      $(systemctl is-active caddy)"
echo "    Domain:          https://$DOMAIN  (waits on Route 53 A record + LE cert)"
