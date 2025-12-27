#!/bin/bash
# Run load tests and collect metrics data

set -e

SCENARIO=${1:-baseline}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="load_test_results/${SCENARIO}_${TIMESTAMP}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Load Test Runner ===${NC}"
echo -e "Scenario: ${YELLOW}${SCENARIO}${NC}"
echo -e "Output: ${YELLOW}${OUTPUT_DIR}${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if services are running
echo -e "${GREEN}Checking services...${NC}"
if ! curl -s http://localhost:8080/healthz > /dev/null; then
    echo -e "${RED}Error: Web service not running at localhost:8080${NC}"
    echo "Run: docker compose up -d"
    exit 1
fi

if ! curl -s http://localhost:9091/-/healthy > /dev/null; then
    echo -e "${RED}Error: Prometheus not running at localhost:9091${NC}"
    echo "Run: docker compose up -d"
    exit 1
fi

echo -e "${GREEN}Services are healthy!${NC}\n"

# Start Prometheus data export in background
echo -e "${GREEN}Starting metrics collection...${NC}"
python3 tools/export_prometheus.py \
    --output "$OUTPUT_DIR/metrics.csv" \
    --interval 15 \
    --prom-url http://localhost:9091 &
EXPORTER_PID=$!

# Give exporter time to start
sleep 2

echo -e "${GREEN}Running $SCENARIO load test...${NC}\n"

case $SCENARIO in
  baseline)
    locust -f tests/load/locustfile.py \
      --headless \
      --users 100 \
      --spawn-rate 10 \
      --run-time 5m \
      --host http://localhost:8080 \
      --html "$OUTPUT_DIR/locust_report.html" \
      --csv "$OUTPUT_DIR/locust_stats"
    ;;

  stress)
    locust -f tests/load/locustfile.py \
      --headless \
      --shape tests.load.scenarios.StressTest \
      --host http://localhost:8080 \
      --html "$OUTPUT_DIR/locust_report.html" \
      --csv "$OUTPUT_DIR/locust_stats"
    ;;

  spike)
    locust -f tests/load/locustfile.py \
      --headless \
      --shape tests.load.scenarios.SpikeTest \
      --host http://localhost:8080 \
      --html "$OUTPUT_DIR/locust_report.html" \
      --csv "$OUTPUT_DIR/locust_stats"
    ;;

  slow-client)
    # Mix of normal and slow clients
    locust -f tests/load/locustfile.py \
      --headless \
      --shape tests.load.scenarios.SlowClientTest \
      --host http://localhost:8080 \
      --html "$OUTPUT_DIR/locust_report.html" \
      --csv "$OUTPUT_DIR/locust_stats" \
      ChatUser:90,SlowChatUser:10
    ;;

  *)
    echo -e "${RED}Unknown scenario: $SCENARIO${NC}"
    echo "Available scenarios: baseline, stress, spike, slow-client"
    kill $EXPORTER_PID
    exit 1
    ;;
esac

# Stop Prometheus export
echo -e "\n${GREEN}Stopping metrics collection...${NC}"
kill -INT $EXPORTER_PID
wait $EXPORTER_PID 2>/dev/null || true

echo -e "\n${GREEN}=== Test Complete ===${NC}"
echo -e "Results saved to: ${YELLOW}${OUTPUT_DIR}${NC}"
echo ""
echo "Files created:"
ls -lh "$OUTPUT_DIR"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. Review Locust report: open $OUTPUT_DIR/locust_report.html"
echo "2. Analyze metrics: python analysis/analyze_results.py $OUTPUT_DIR/metrics.csv"
