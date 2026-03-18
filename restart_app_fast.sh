#!/usr/bin/env bash
set -e

docker-compose up -d --no-deps --force-recreate app

# wait up to 60s for app to be ready
for i in {1..60}; do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/metrics || true)
  if [ "$code" = "200" ]; then
    break
  fi
  sleep 1
done

./health_gate.sh
