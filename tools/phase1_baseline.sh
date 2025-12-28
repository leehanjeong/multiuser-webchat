#!/bin/bash
# Phase 1: Baseline - Normal Operation
# Duration: 5 minutes
# Purpose: Establish performance baseline

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="load_test_results/phase1_baseline_${TIMESTAMP}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Phase 1: Baseline (5 min)            ${NC}"
echo -e "${BLUE}========================================${NC}"
echo "100 users, ChatUser (0.3 msg/sec)"
echo "Total: ~30 msg/sec"
echo ""

mkdir -p "$OUTPUT_DIR"

# Start metrics collection
echo -e "${GREEN}Starting metrics collection...${NC}"
poetry run python tools/export_prometheus.py \
    --output "$OUTPUT_DIR/metrics.csv" \
    --interval 2 \
    --prom-url http://localhost:9091 &
EXPORTER_PID=$!

sleep 3

# Run test
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 5m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/baseline.html" \
    --csv "$OUTPUT_DIR/baseline" \
    ChatUser

# Stop metrics collection
echo -e "\n${GREEN}Stopping metrics collection...${NC}"
kill -INT $EXPORTER_PID
wait $EXPORTER_PID 2>/dev/null || true

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Phase 1 Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Results: $OUTPUT_DIR"
