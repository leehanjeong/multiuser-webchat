#!/bin/bash
# Phase 6: High Churn Test - Connection Overhead
# Duration: 5 minutes
# Purpose: Test system's ability to handle frequent reconnections

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="load_test_results/phase6_high_churn_${TIMESTAMP}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Phase 6: High Churn (5 min)          ${NC}"
echo -e "${BLUE}========================================${NC}"
echo "50 users reconnecting every 5 seconds"
echo "Goal: ~10 new connections/sec sustained"
echo "Tests: Connection overhead, cleanup, memory leaks"
echo ""

mkdir -p "$OUTPUT_DIR"

# Start metrics collection
echo -e "${GREEN}Starting metrics collection...${NC}"
poetry run python tools/export_prometheus.py \
    --output "$OUTPUT_DIR/metrics.csv" \
    --interval 2 \
    --prom-url http://localhost:9091 &
EXPORTER_PID=$!

sleep 2

# Run test with churning users
echo -e "\n${YELLOW}Starting high churn test...${NC}"
echo "Each user connects, waits 5s, disconnects, repeats"
echo "Expected: ~10 connections/sec + ~10 disconnections/sec"

poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 50 \
    --spawn-rate 10 \
    --run-time 5m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/high_churn.html" \
    --csv "$OUTPUT_DIR/high_churn" \
    ChurnUser

# Stop metrics collection
echo -e "\n${GREEN}Stopping metrics collection...${NC}"
kill -INT $EXPORTER_PID
wait $EXPORTER_PID 2>/dev/null || true

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Phase 6 Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Results: $OUTPUT_DIR"
echo ""
echo "Analysis checklist:"
echo "  - Check connection_rate and disconnection_rate metrics"
echo "  - Monitor memory growth (potential connection leaks?)"
echo "  - Verify WebSocket cleanup is working"
echo "  - Check eventloop_lag (connection overhead impact)"
