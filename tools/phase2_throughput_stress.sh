#!/bin/bash
# Phase 2: Throughput Stress Test - Message Rate Escalation
# Duration: 8 minutes (4 levels × 2 min each)
# Purpose: Find message throughput bottleneck (typically Redis)

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="load_test_results/phase2_throughput_stress_${TIMESTAMP}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Phase 2: Throughput Stress (8 min)   ${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Variable: msg/sec per user"
echo "Fixed: 100 users, small messages"
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

# Level 1: Moderate (0.7 msg/sec × 100 = 70 msg/s total)
echo -e "\n${GREEN}Level 1: Moderate Load (2 min)${NC}"
echo "100 users × 0.7 msg/sec = 70 msg/s"
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/level1_moderate.html" \
    --csv "$OUTPUT_DIR/level1_moderate" \
    ModerateUser || true

sleep 10

# Level 2: High (2.0 msg/sec × 100 = 200 msg/s total)
echo -e "\n${YELLOW}Level 2: High Load (2 min)${NC}"
echo "100 users × 2.0 msg/sec = 200 msg/s"
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/level2_high.html" \
    --csv "$OUTPUT_DIR/level2_high" \
    ActiveUser || true

sleep 10

# Level 3: Extreme (5.0 msg/sec × 100 = 500 msg/s total)
echo -e "\n${YELLOW}Level 3: Extreme Load (2 min)${NC}"
echo "100 users × 5.0 msg/sec = 500 msg/s"
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/level3_extreme.html" \
    --csv "$OUTPUT_DIR/level3_extreme" \
    MachineGunUser || true

sleep 10

# Level 4: INSANE (15.0 msg/sec × 100 = 1500 msg/s total!)
echo -e "\n${RED}Level 4: INSANE Load (2 min) - EXPECT FAILURE!${NC}"
echo "100 users × 15.0 msg/sec = 1500 msg/s!"
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/level4_insane.html" \
    --csv "$OUTPUT_DIR/level4_insane" \
    SuperMachineGunUser || true

# Stop metrics collection
echo -e "\n${GREEN}Stopping metrics collection...${NC}"
kill -INT $EXPORTER_PID
wait $EXPORTER_PID 2>/dev/null || true

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Phase 2 Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Results: $OUTPUT_DIR"
echo ""
echo "Message rate progression:"
echo "  Level 1: 70 msg/s (ModerateUser)"
echo "  Level 2: 200 msg/s (ActiveUser)"
echo "  Level 3: 500 msg/s (MachineGunUser)"
echo "  Level 4: 1500 msg/s (SuperMachineGunUser!)"
