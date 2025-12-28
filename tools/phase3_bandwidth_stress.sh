#!/bin/bash
# Phase 3: Bandwidth Stress Test - Message Size Escalation
# Duration: 8 minutes (4 levels × 2 min each)
# Purpose: Find network bandwidth limits

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="load_test_results/phase3_bandwidth_stress_${TIMESTAMP}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Phase 3: Bandwidth Stress (8 min)    ${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Variable: message size"
echo "Fixed: 100 users, HeavyLifterUser"
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

# Level 1: 10 KB messages
echo -e "\n${GREEN}Level 1: 10 KB messages (2 min)${NC}"
echo "100 users × 10 KB = ~1 MB/s"
MESSAGE_SIZE=10240 poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/level1_10kb.html" \
    --csv "$OUTPUT_DIR/level1_10kb" \
    HeavyLifterUser || true

sleep 10

# Level 2: 50 KB messages
echo -e "\n${YELLOW}Level 2: 50 KB messages (2 min)${NC}"
echo "100 users × 50 KB = ~5 MB/s"
MESSAGE_SIZE=51200 poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/level2_50kb.html" \
    --csv "$OUTPUT_DIR/level2_50kb" \
    HeavyLifterUser || true

sleep 10

# Level 3: 100 KB messages
echo -e "\n${YELLOW}Level 3: 100 KB messages (2 min)${NC}"
echo "100 users × 100 KB = ~10 MB/s"
MESSAGE_SIZE=102400 poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/level3_100kb.html" \
    --csv "$OUTPUT_DIR/level3_100kb" \
    HeavyLifterUser || true

sleep 10

# Level 4: 500 KB messages
echo -e "\n${RED}Level 4: 500 KB messages (2 min) - EXPECT FAILURE!${NC}"
echo "100 users × 500 KB = ~50 MB/s!"
MESSAGE_SIZE=512000 poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/level4_500kb.html" \
    --csv "$OUTPUT_DIR/level4_500kb" \
    HeavyLifterUser || true

sleep 10

# Level 5: 1000 KB messages (approx 1 MB)
echo -e "\n${RED}Level 5: 1000 KB messages (2 min) - EXPECT FAILURE!${NC}"
echo "100 users × 1000 KB = ~100 MB/s!"
MESSAGE_SIZE=1024000 poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/level5_1000kb.html" \
    --csv "$OUTPUT_DIR/level5_1000kb" \
    HeavyLifterUser || true

# Stop metrics collection
echo -e "\n${GREEN}Stopping metrics collection...${NC}"
kill -INT $EXPORTER_PID
wait $EXPORTER_PID 2>/dev/null || true

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Phase 3 Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Results: $OUTPUT_DIR"
echo ""
echo "Message size progression:"
echo "  Level 1: 10 KB (~1 MB/s)"
echo "  Level 2: 50 KB (~5 MB/s)"
echo "  Level 3: 150 KB (~15 MB/s)"
echo "  Level 4: 300 KB (~30 MB/s!)"
