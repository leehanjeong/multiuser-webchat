#!/bin/bash
# Comprehensive 8-Phase Load Test
# Tests different failure modes: CPU, Bandwidth, Connections, Bad Actor, Churn

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="load_test_results/comprehensive_${TIMESTAMP}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to check and restart services if needed
check_and_restart_services() {
    echo -e "${YELLOW}Checking services...${NC}"

    # Check web service
    if ! curl -s http://localhost:8080/healthz > /dev/null 2>&1; then
        echo -e "${RED}Web service not responding! Restarting...${NC}"
        docker compose restart web
        sleep 10

        if ! curl -s http://localhost:8080/healthz > /dev/null 2>&1; then
            echo -e "${RED}Web service still not responding after restart!${NC}"
            return 1
        fi
    fi

    echo -e "${GREEN}Services OK${NC}"
    return 0
}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Comprehensive 8-Phase Load Test      ${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Output: ${YELLOW}${OUTPUT_DIR}${NC}\n"

mkdir -p "$OUTPUT_DIR"

# Initial service check
echo -e "${GREEN}Checking initial services...${NC}"
if ! curl -s http://localhost:8080/healthz > /dev/null; then
    echo -e "${RED}Error: Web service not running${NC}"
    echo "Run: docker compose up -d"
    exit 1
fi

if ! curl -s http://localhost:9091/-/healthy > /dev/null; then
    echo -e "${RED}Error: Prometheus not running${NC}"
    echo "Run: docker compose up -d"
    exit 1
fi

echo -e "${GREEN}Services are healthy!${NC}\n"

# Start metrics collection (2-second interval)
echo -e "${GREEN}Starting metrics collection (2-second interval)...${NC}"
poetry run python tools/export_prometheus.py \
    --output "$OUTPUT_DIR/metrics.csv" \
    --interval 2 \
    --prom-url http://localhost:9091 &
EXPORTER_PID=$!

sleep 3

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Phase 1: Baseline (5 min)            ${NC}"
echo -e "${BLUE}========================================${NC}"
echo "100 users, ChatUser (0.3 msg/sec), Total: 30 msg/sec"
check_and_restart_services || true
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 5m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase1_baseline.html" \
    --csv "$OUTPUT_DIR/phase1_baseline" \
    ChatUser || true

sleep 2

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Phase 2: The Machine Gun (10 min)    ${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\n${GREEN}Phase 2A: Normal rate (3 min)${NC}"
echo "100 users, ChatUser (0.3 msg/sec), Total: 30 msg/sec"
check_and_restart_services || true
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 3m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase2a_normal.html" \
    --csv "$OUTPUT_DIR/phase2a_normal" \
    ChatUser || true

sleep 2

echo -e "\n${GREEN}Phase 2B: Moderate rate (3 min)${NC}"
echo "100 users, ModerateUser (0.7 msg/sec), Total: 70 msg/sec"
check_and_restart_services || true
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 3m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase2b_moderate.html" \
    --csv "$OUTPUT_DIR/phase2b_moderate" \
    ModerateUser || true

sleep 2

echo -e "\n${YELLOW}Phase 2C: High rate (2 min)${NC}"
echo "100 users, ActiveUser (2.0 msg/sec), Total: 200 msg/sec"
check_and_restart_services || true
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase2c_high.html" \
    --csv "$OUTPUT_DIR/phase2c_high" \
    ActiveUser || true

sleep 2

echo -e "\n${RED}Phase 2D: EXTREME rate (2 min) - EXPECT STRESS!${NC}"
echo "100 users, MachineGunUser (5.0 msg/sec), Total: 500 msg/sec!"
check_and_restart_services || true
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase2d_extreme.html" \
    --csv "$OUTPUT_DIR/phase2d_extreme" \
    MachineGunUser || true

sleep 2

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Phase 3: The Heavy Lifter (10 min)   ${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\n${GREEN}Phase 3A: 1 KB messages (3 min)${NC}"
echo "100 users, HeavyLifterUser, 1 KB"
check_and_restart_services || true
MESSAGE_SIZE=1024 poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 3m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase3a_1kb.html" \
    --csv "$OUTPUT_DIR/phase3a_1kb" \
    HeavyLifterUser || true

sleep 2

echo -e "\n${GREEN}Phase 3B: 10 KB messages (3 min)${NC}"
echo "100 users, HeavyLifterUser, 10 KB"
check_and_restart_services || true
MESSAGE_SIZE=10240 poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 3m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase3b_10kb.html" \
    --csv "$OUTPUT_DIR/phase3b_10kb" \
    HeavyLifterUser || true

sleep 2

echo -e "\n${YELLOW}Phase 3C: 50 KB messages (2 min)${NC}"
echo "100 users, HeavyLifterUser, 50 KB"
check_and_restart_services || true
MESSAGE_SIZE=51200 poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase3c_50kb.html" \
    --csv "$OUTPUT_DIR/phase3c_50kb" \
    HeavyLifterUser || true

sleep 2

echo -e "\n${RED}Phase 3D: 100 KB messages (2 min) - EXPECT STRESS!${NC}"
echo "100 users, HeavyLifterUser, 100 KB"
check_and_restart_services || true
MESSAGE_SIZE=102400 poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 2m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase3d_100kb.html" \
    --csv "$OUTPUT_DIR/phase3d_100kb" \
    HeavyLifterUser || true

sleep 2

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Phase 4: The Crowd (5 min)           ${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\n${GREEN}Phase 4A: 100 users (1 min)${NC}"
check_and_restart_services || true
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 1m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase4a_100users.html" \
    --csv "$OUTPUT_DIR/phase4a_100users" \
    ChatUser || true

sleep 2

echo -e "\n${GREEN}Phase 4B: 200 users (1 min)${NC}"
check_and_restart_services || true
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 200 \
    --spawn-rate 20 \
    --run-time 1m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase4b_200users.html" \
    --csv "$OUTPUT_DIR/phase4b_200users" \
    ChatUser || true

sleep 2

echo -e "\n${YELLOW}Phase 4C: 270 users (1 min)${NC}"
check_and_restart_services || true
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 270 \
    --spawn-rate 20 \
    --run-time 1m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase4c_270users.html" \
    --csv "$OUTPUT_DIR/phase4c_270users" \
    ChatUser || true

sleep 2

echo -e "\n${YELLOW}Phase 4D: 300 users (1 min)${NC}"
check_and_restart_services || true
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 300 \
    --spawn-rate 20 \
    --run-time 1m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase4d_300users.html" \
    --csv "$OUTPUT_DIR/phase4d_300users" \
    ChatUser || true

sleep 2

echo -e "\n${RED}Phase 4E: 320 users (1 min) - BREAKING POINT!${NC}"
check_and_restart_services || true
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 320 \
    --spawn-rate 20 \
    --run-time 1m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase4e_320users.html" \
    --csv "$OUTPUT_DIR/phase4e_320users" \
    ChatUser || true

sleep 2

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Phase 5: Cool Down & Recovery (4 min)${NC}"
echo -e "${BLUE}========================================${NC}"
echo "320 → 100 users (rapid decrease)"
check_and_restart_services || true
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 55 \
    --run-time 4m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase5_recovery.html" \
    --csv "$OUTPUT_DIR/phase5_recovery" \
    ChatUser || true

sleep 2

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Phase 6A: The One Bad Actor (5 min)  ${NC}"
echo -e "${BLUE}========================================${NC}"
echo "100 normal users + 1 bad actor (5 msg/sec × 100 KB)"
check_and_restart_services || true
BAD_ACTOR_SIZE=102400 poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 101 \
    --spawn-rate 20 \
    --run-time 5m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase6a_bad_actor.html" \
    --csv "$OUTPUT_DIR/phase6a_bad_actor" \
    ChatUser:100 BadActor:1 || true

sleep 2

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Phase 6B: High Churn Rate (5 min)    ${NC}"
echo -e "${BLUE}========================================${NC}"
echo "50 users, 10 connections/sec churn"
check_and_restart_services || true
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 50 \
    --spawn-rate 10 \
    --run-time 5m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase6b_high_churn.html" \
    --csv "$OUTPUT_DIR/phase6b_high_churn" \
    ChurnUser || true

sleep 2

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Phase 7: Final Baseline (1 min)      ${NC}"
echo -e "${BLUE}========================================${NC}"
echo "100 users, ChatUser (verify full recovery)"
check_and_restart_services || true
poetry run locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 20 \
    --run-time 1m \
    --host http://localhost:8080 \
    --html "$OUTPUT_DIR/phase7_final_baseline.html" \
    --csv "$OUTPUT_DIR/phase7_final_baseline" \
    ChatUser || true

# Stop metrics collection
echo -e "\n${GREEN}Stopping metrics collection...${NC}"
kill -INT $EXPORTER_PID
wait $EXPORTER_PID 2>/dev/null || true

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}      Test Complete!                    ${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Results saved to: ${YELLOW}${OUTPUT_DIR}${NC}\n"

# Show metrics file info
if [ -f "$OUTPUT_DIR/metrics.csv" ]; then
    LINES=$(wc -l < "$OUTPUT_DIR/metrics.csv")
    SIZE=$(du -h "$OUTPUT_DIR/metrics.csv" | cut -f1)
    echo -e "${GREEN}Metrics collected:${NC}"
    echo "  - Rows: $LINES"
    echo "  - Size: $SIZE"
    echo "  - File: metrics.csv"
fi

echo -e "\n${GREEN}Test Summary:${NC}"
echo "  - Total duration: ~47 minutes"
echo "  - 8 phases completed (with auto-recovery)"
echo "  - Expected data points: ~1,410 (2-second interval)"
echo ""
echo -e "${GREEN}Phase breakdown:${NC}"
echo "  1. Baseline: Normal operation"
echo "  2. Machine Gun: CPU stress (0.3→5.0 msg/sec)"
echo "  3. Heavy Lifter: Bandwidth stress (1KB→100KB)"
echo "  4. The Crowd: Connection stress (100→320 users)"
echo "  5. Recovery: Self-healing verification"
echo "  6A. Bad Actor: 1 malicious user isolation test"
echo "  6B. High Churn: Connection overhead test"
echo "  7. Final Baseline: Full recovery verification"

echo -e "\n${GREEN}Next steps:${NC}"
echo "1. Review phase reports: open $OUTPUT_DIR/*.html"
echo "2. Analyze metrics: poetry run jupyter notebook"
echo "3. Load data: pd.read_csv('$OUTPUT_DIR/metrics.csv')"
echo "4. Look for failure patterns in phases 2D, 3D, 4E"
