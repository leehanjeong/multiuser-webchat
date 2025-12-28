#!/bin/bash
# Phase 5: Bad Actor Test - Malicious User Isolation
# Duration: 5 minutes
# Purpose: Test system's ability to handle one malicious heavy user

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="load_test_results/phase5_bad_actor_${TIMESTAMP}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Phase 5: Bad Actor (5 min)           ${NC}"
echo -e "${BLUE}========================================${NC}"
echo "100 normal users + 1 bad actor"
echo "Bad actor: 15 msg/sec × 200 KB"
echo "Goal: Ensure bad actor doesn't degrade normal users"
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

# Run test with 100 normal + 1 bad actor
echo -e "\n${RED}Injecting bad actor...${NC}"
echo "Normal users: 100 × ChatUser (0.3 msg/sec, small)"
echo "Bad actor: 1 × BadActor (15 msg/sec, 200 KB)"
echo "Total load: ~30 normal msg/s + 15 heavy msg/s"

BAD_ACTOR_SIZE=204800 poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 101 \
    --spawn-rate 20 \
    --run-time 5m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/bad_actor.html" \
    --csv "$OUTPUT_DIR/bad_actor" \
    ChatUser:100 BadActor:1

# Stop metrics collection
echo -e "\n${GREEN}Stopping metrics collection...${NC}"
kill -INT $EXPORTER_PID
wait $EXPORTER_PID 2>/dev/null || true

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Phase 5 Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Results: $OUTPUT_DIR"
echo ""
echo "Analysis checklist:"
echo "  - Did normal users experience degraded latency?"
echo "  - Was the bad actor isolated or did it affect everyone?"
echo "  - Check memory/CPU spikes during bad actor activity"
