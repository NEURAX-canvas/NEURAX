#!/usr/bin/env bash
# ─── NEURAX Health Check Script ───────────────────────────────────────────────
# Verifies that all three NEURAX services are running and healthy.
#
# Usage: ./healthcheck.sh
# ──────────────────────────────────────────────────────────────────────────────

set -uo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Service definitions: name|port|health_endpoint
SERVICES=(
    "neurax-service|9098|/health"
    "neurax-agent|8099|/health"
    "neurax-ui|8081|"
)

ALL_HEALTHY=true

for service_def in "${SERVICES[@]}"; do
    IFS='|' read -r name port endpoint <<< "$service_def"

    if [ -z "$endpoint" ]; then
        # No health endpoint (e.g., frontend) — just check if port is open
        if curl -s --connect-timeout 3 "http://localhost:${port}" >/dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} ${name} (port ${port}): healthy"
        else
            echo -e "${RED}✗${NC} ${name} (port ${port}): NOT reachable"
            ALL_HEALTHY=false
        fi
    else
        # Check health endpoint
        response=$(curl -s --connect-timeout 3 "http://localhost:${port}${endpoint}" 2>/dev/null)
        if echo "$response" | grep -q '"ok"' || echo "$response" | grep -q '"status"'; then
            echo -e "${GREEN}✓${NC} ${name} (port ${port}): healthy"
        else
            echo -e "${RED}✗${NC} ${name} (port ${port}): NOT healthy"
            ALL_HEALTHY=false
        fi
    fi
done

echo ""

if [ "$ALL_HEALTHY" = true ]; then
    echo -e "${GREEN}All services are healthy!${NC}"
    exit 0
else
    echo -e "${RED}Some services are not healthy. Check the logs for details.${NC}"
    exit 1
fi
