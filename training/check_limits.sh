#!/bin/bash
# Check current job status and limits

echo "=== Current Jobs in Queue ==="
squeue -u u5514611

echo ""
echo "=== Recent Job History ==="
sacct --format=JobID,JobName,State,Submit,Start,End -S $(date -d '1 day ago' +%Y-%m-%d) | head -20

echo ""
echo "=== Association Limits ==="
sacctmgr show assoc where user=u5514611 format=user,account,partition,maxjobs,maxsubmit

echo ""
echo "=== QOS Limits ==="
sacctmgr show qos format=name,maxjobspu,maxsubmitpu
