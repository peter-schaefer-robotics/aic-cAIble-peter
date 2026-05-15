#!/usr/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

# Resolve script directory (works whether sourced or executed) so the zenoh
# config path is host-independent.
_AIC_REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# No auth — AIC_ENABLE_ACL is disabled in docker-compose
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'

# Only point at the session config if it actually exists at the resolved
# location. Inside the submission docker image the `docker/aic_eval/`
# tree isn't copied — the container's own entrypoint sets
# ZENOH_CONFIG_OVERRIDE with the router endpoint instead, so leaving
# ZENOH_SESSION_CONFIG_URI unset is correct there.
if [ -f "${_AIC_REPO_ROOT}/docker/aic_eval/aic_zenoh_config.json5" ]; then
    export ZENOH_SESSION_CONFIG_URI="${_AIC_REPO_ROOT}/docker/aic_eval/aic_zenoh_config.json5"
fi
