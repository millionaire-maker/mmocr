#!/usr/bin/env bash
# 每隔固定时间自动提交并推送当前仓库，提交信息为 push 当下的时间。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# 可通过环境变量覆盖：AUTO_PUSH_INTERVAL（秒）、AUTO_PUSH_REMOTE、AUTO_PUSH_BRANCH
INTERVAL="${AUTO_PUSH_INTERVAL:-1800}"
REMOTE="${AUTO_PUSH_REMOTE:-origin}"
BRANCH="${AUTO_PUSH_BRANCH:-master}"

while true; do
  changes_pending=false

  # 工作区或索引有变更
  if ! git diff --quiet --ignore-submodules; then
    changes_pending=true
  fi

  # 还有未跟踪文件
  if [ -n "$(git ls-files --others --exclude-standard)" ]; then
    changes_pending=true
  fi

  if [ "$changes_pending" = true ]; then
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    git add -A

    # 确认有实际内容后再提交
    if ! git diff --cached --quiet --ignore-submodules; then
      git commit -m "auto push ${ts}"
      if ! git push "$REMOTE" "$BRANCH"; then
        echo "[auto-push] $(date '+%Y-%m-%d %H:%M:%S') push failed，稍后重试。" >&2
      fi
    fi
  fi

  sleep "$INTERVAL"
done
