#!/usr/bin/env bash
set -e
curl -s http://localhost:8000/health/live && echo
curl -s http://localhost:8000/health/ready && echo
curl -s -o /dev/null -w "metrics_status=%{http_code}\n" http://localhost:8000/metrics
