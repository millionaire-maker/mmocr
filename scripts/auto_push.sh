#!/usr/bin/env bash
# 每隔固定时间自动提交并推送当前仓库，提交信息为 push 当下的时间。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# 可通过环境变量覆盖：
# - AUTO_PUSH_INTERVAL（秒）
# - AUTO_PUSH_REMOTE / AUTO_PUSH_BRANCH
# - AUTO_PUSH_LOCK_FILE
# - AUTO_PUSH_PUSH_TIMEOUT（秒）
INTERVAL="${AUTO_PUSH_INTERVAL:-1800}"
REMOTE="${AUTO_PUSH_REMOTE:-origin}"
BRANCH="${AUTO_PUSH_BRANCH:-master}"
LOCK_FILE="${AUTO_PUSH_LOCK_FILE:-/tmp/mmocr_auto_push.lock}"
PUSH_TIMEOUT="${AUTO_PUSH_PUSH_TIMEOUT:-7200}"

log() {
  echo "[auto-push] $(date '+%Y-%m-%d %H:%M:%S') $*"
}

# 防止重复启动多个实例导致相互抢占/卡住
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "检测到已有 auto_push 实例在运行（lock: $LOCK_FILE），本实例退出。"
  exit 0
fi

# 禁止交互式提示，避免 nohup 下卡住
export GIT_TERMINAL_PROMPT=0
export GIT_LFS_FORCE_PROGRESS=1
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6}"

while true; do
  changes_pending=false

  if [ -n "$(git status --porcelain=v1 --untracked-files=normal)" ]; then
    changes_pending=true
  fi

  if [ "$changes_pending" = true ]; then
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    log "检测到变更，准备提交（commit msg time: $ts）"

    if ! git add -A; then
      log "git add 失败，稍后重试。"
      sleep "$INTERVAL"
      continue
    fi

    # 确认有实际内容后再提交
    if ! git diff --cached --quiet --ignore-submodules; then
      if ! git commit -m "auto push ${ts}"; then
        log "git commit 失败（可能有冲突/锁/权限），稍后重试。"
        sleep "$INTERVAL"
        continue
      fi
    fi
  fi

  # 如果本地领先远端，则尝试推送（包含 LFS 权重上传，可能耗时较长）
  ahead_count=0
  if git rev-parse --verify "$REMOTE/$BRANCH" >/dev/null 2>&1; then
    ahead_count="$(git rev-list --count "$REMOTE/$BRANCH..HEAD" 2>/dev/null || echo 0)"
  else
    ahead_count=1
  fi

  if [ "${ahead_count}" != "0" ]; then
    log "开始推送到 $REMOTE/$BRANCH（ahead: ${ahead_count}，timeout: ${PUSH_TIMEOUT}s）"
    if timeout "$PUSH_TIMEOUT" git push --progress "$REMOTE" "$BRANCH"; then
      log "推送成功。"
    else
      rc=$?
      log "推送失败（rc=$rc），稍后自动重试。"
    fi
  fi

  sleep "$INTERVAL"
done
