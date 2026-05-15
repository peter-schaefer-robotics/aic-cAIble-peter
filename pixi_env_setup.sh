#!/usr/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

# Resolve script directory (works whether sourced or executed) so the zenoh
# config path is host-independent.
_AIC_REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# No auth — AIC_ENABLE_ACL is disabled in docker-compose
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
export ZENOH_SESSION_CONFIG_URI="${_AIC_REPO_ROOT}/docker/aic_eval/aic_zenoh_config.json5"
