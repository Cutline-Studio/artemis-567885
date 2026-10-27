#!/bin/bash
# on-instance watchdog: destroy THIS instance at rent+60 min no matter what (instance-scoped key from Vast)
R=$(cat /w/R); END=$(( R + 60*60 ))
CID=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_ID=//p')
K=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_API_KEY=//p')
while [ $(date +%s) -lt $END ]; do sleep 20; done
for i in 1 2 3 4 5; do curl -s -X DELETE -H "Authorization: Bearer $K" https://console.vast.ai/api/v0/instances/$CID/ > /root/selfdestroy.log 2>&1 && break; sleep 10; done
