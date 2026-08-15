#!/bin/bash
# ── VIRAL Bot Oracle Cloud 1-Line Installer ────────────────────────────────────
# Run this script on your Oracle Cloud Ubuntu VPS instance to launch 24/7.

echo "🚀 Setting up VIRAL Bot on Oracle Cloud Instance..."

# 1. Update system & install Docker + docker-compose
sudo apt-get update -y
sudo apt-get install -y docker.io docker-compose curl git tar

# 2. Enable & start Docker service
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER

# 3. Extract VIRAL bundle if tar.gz exists
if [ -f "viral_cloud_bundle.tar.gz" ]; then
    tar -xzf viral_cloud_bundle.tar.gz
fi

# 4. Launch all 4 Docker containers 24/7 in background
sudo docker-compose up -d --force-recreate

echo ""
echo "🎉 SUCCESS! Your VIRAL bot is now running 24/7/365 on Oracle Cloud!"
echo "📱 You can close your laptop lid anytime — Telegram bot is live in the cloud!"
