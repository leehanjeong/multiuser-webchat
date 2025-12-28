#!/bin/bash
# Phase 4: Connection Stress Test - Concurrent Users Escalation
# Duration: 10 minutes (5 levels × 2 min each)
# Purpose: Find maximum concurrent connection capacity

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="load_test_results/phase4_connection_stress_${TIMESTAMP}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Phase 4: Connection Stress (10 min)  ${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Variable: concurrent users"
echo "Fixed: ChatUser, normal message rate"
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

# Level 1: 100 users (baseline)
echo -e "\n${GREEN}Level 1: 100 users (2 min)${NC}"
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/level1_100users.html" \
    --csv "$OUTPUT_DIR/level1_100users" \
    ChatUser || true

sleep 10

# Level 2: 200 users
echo -e "\n${GREEN}Level 2: 200 users (2 min)${NC}"
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 200 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/level2_200users.html" \
    --csv "$OUTPUT_DIR/level2_200users" \
    ChatUser || true

sleep 10

# Level 3: 300 users
echo -e "\n${YELLOW}Level 3: 300 users (2 min)${NC}"
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 300 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/level3_350users.html" \
    --csv "$OUTPUT_DIR/level3_350users" \
    ChatUser || true

sleep 10

# Level 4: 400 users (near breaking point)
echo -e "\n${YELLOW}Level 4: 400 users (2 min) - NEAR LIMIT!${NC}"
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 400 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/level4_450users.html" \
    --csv "$OUTPUT_DIR/level4_450users" \
    ChatUser || true

sleep 10

# Level 5: 500 users (breaking point!)
echo -e "\n${RED}Level 5: 500 users (2 min) - BREAKING POINT!${NC}"
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 500 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/level5_500users.html" \
    --csv "$OUTPUT_DIR/level5_500users" \
    ChatUser || true

sleep 10

# Level 6: 600 users (breaking point!)
echo -e "\n${RED}Level 6: 600 users (2 min) - BREAKING POINT!${NC}"
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 600 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/level5_500users.html" \
    --csv "$OUTPUT_DIR/level5_500users" \
    ChatUser || true

# Stop metrics collection
echo -e "\n${GREEN}Stopping metrics collection...${NC}"
kill -INT $EXPORTER_PID
wait $EXPORTER_PID 2>/dev/null || true

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Phase 4 Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Results: $OUTPUT_DIR"
echo ""
echo "Concurrent users progression:"
echo "  Level 1: 100 users"
echo "  Level 2: 200 users"
echo "  Level 3: 350 users"
echo "  Level 4: 450 users"
echo "  Level 5: 500 users (breaking point expected!)"
