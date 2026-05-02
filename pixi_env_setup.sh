#!/usr/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

# No auth — AIC_ENABLE_ACL is disabled in docker-compose
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
export ZENOH_SESSION_CONFIG_URI="/home/richard/aic/docker/aic_eval/aic_zenoh_config.json5"
