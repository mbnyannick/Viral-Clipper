#!/bin/bash
# ── VIRAL 24/7 Cloud Deployment Bundle Generator ──────────────────────────────
# Run this script to pack your VIRAL bot into a single uploadable cloud archive.

echo "📦 Bundling VIRAL Bot for 24/7 Cloud Server Deployment..."

TAR_FILE="viral_cloud_bundle.tar.gz"

tar -czf "$TAR_FILE" \
  --exclude="tmp/*" \
  --exclude="logs/*" \
  --exclude=".git" \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  Dockerfile \
  docker-compose.yml \
  requirements.txt \
  approved_users.json \
  cookies.txt \
  .env \
  bot/ \
  pipeline/ \
  config/ \
  assets/

echo "✅ Created 24/7 Cloud Bundle: $TAR_FILE"
echo ""
echo "🚀 3 STEPS TO RUN 24/7 IN THE CLOUD:"
echo "1. Upload $TAR_FILE to your Linux VPS (e.g. DigitalOcean, Hetzner, AWS, Railway)"
echo "2. Run: tar -xzf $TAR_FILE"
echo "3. Run: docker-compose up -d"
echo ""
echo "🎉 Once done, your bot will run 24/7 even when your laptop is closed!"
