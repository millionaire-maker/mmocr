#!/usr/bin/env bash
# 自动从远程训练服务器同步数据到本机仓库（本机为主）。
#
# 设计目标：
# - 使用 rsync --update：仅当远程文件更新时间更“新”时才覆盖本地；本地较新则保留本地。
# - work_dirs/ 单独增量同步（适合训练日志/ckpt）。
# - 代码/配置等通过远程 git 变更列表（diff + untracked）精确同步。
# - 可选：同步后在远程执行 reset/clean，清空远程 git 变更（不产生额外提交历史）。
#
# 安全提示：
# - 不要把 SSH 密码写进脚本。建议先配置 SSH 公钥免密登录（ssh-copy-id）。
#
# 使用示例（前台跑一次）：
#   SYNC_ONCE=1 ./scripts/auto_sync_remote.sh
#
# 后台常驻：
#   nohup SYNC_INTERVAL=600 ./scripts/auto_sync_remote.sh >/tmp/mmocr_auto_sync_remote.log 2>&1 &
#
# 环境变量（可覆盖默认）：
# - SYNC_REMOTE_HOST / SYNC_REMOTE_USER / SYNC_REMOTE_PORT
# - SYNC_REMOTE_REPO_ROOT（远端 mmocr 仓库路径）
# - SYNC_INTERVAL（秒），SYNC_ONCE=1 表示只同步一次
# - SYNC_WORKDIRS=1/0，SYNC_GIT_CHANGED=1/0
# - SYNC_REMOTE_RESET=1/0（同步后远端 reset --hard 到 origin/<branch>）
# - SYNC_REMOTE_CLEAN_UNTRACKED=1/0（同步后远端 git clean -fd；默认 0 更安全）
# - SYNC_REMOTE_KEEP_WORKDIRS=1/0（配合 clean，是否保留 work_dirs；默认 1）
# - SYNC_DRY_RUN=1（只打印 rsync 计划，不真正写入）
# - SYNC_SSH_IDENTITY_FILE（指定私钥路径）
# - SYNC_LOCK_FILE（本地锁文件）
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

INTERVAL="${SYNC_INTERVAL:-1800}"
ONCE="${SYNC_ONCE:-0}"

REMOTE_HOST="${SYNC_REMOTE_HOST:-link.lanyun.net}"
REMOTE_USER="${SYNC_REMOTE_USER:-root}"
REMOTE_PORT="${SYNC_REMOTE_PORT:-44165}"
REMOTE_REPO_ROOT="${SYNC_REMOTE_REPO_ROOT:-/root/lanyun-tmp/mmocr}"

SYNC_WORKDIRS="${SYNC_WORKDIRS:-1}"
SYNC_GIT_CHANGED="${SYNC_GIT_CHANGED:-1}"

REMOTE_RESET="${SYNC_REMOTE_RESET:-1}"
REMOTE_CLEAN_UNTRACKED="${SYNC_REMOTE_CLEAN_UNTRACKED:-0}"
REMOTE_KEEP_WORKDIRS="${SYNC_REMOTE_KEEP_WORKDIRS:-1}"

DRY_RUN="${SYNC_DRY_RUN:-0}"
LOCK_FILE="${SYNC_LOCK_FILE:-/tmp/mmocr_auto_sync_remote.lock}"

SSH_IDENTITY_FILE="${SYNC_SSH_IDENTITY_FILE:-}"
SSH_CONNECT_TIMEOUT="${SYNC_SSH_CONNECT_TIMEOUT:-15}"
REMOTE_CMD_TIMEOUT="${SYNC_REMOTE_CMD_TIMEOUT:-120}"
RSYNC_TIMEOUT="${SYNC_RSYNC_TIMEOUT:-7200}"

log() { echo "[auto-sync-remote] $(date '+%Y-%m-%d %H:%M:%S') $*"; }

# 防止重复启动多个实例导致相互抢占/卡住
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "检测到已有 auto_sync_remote 实例在运行（lock: $LOCK_FILE），本实例退出。"
  exit 0
fi

remote_target="${REMOTE_USER}@${REMOTE_HOST}"

ssh_base=(
  ssh -p "$REMOTE_PORT"
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout="$SSH_CONNECT_TIMEOUT"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
)

if [ -n "$SSH_IDENTITY_FILE" ]; then
  ssh_base+=(-i "$SSH_IDENTITY_FILE")
fi

remote_exec() {
  local cmd="$1"
  timeout "$REMOTE_CMD_TIMEOUT" "${ssh_base[@]}" "$remote_target" /bin/bash -lc "$cmd"
}

rsync_rsh="ssh -p ${REMOTE_PORT} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=${SSH_CONNECT_TIMEOUT} -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
if [ -n "$SSH_IDENTITY_FILE" ]; then
  rsync_rsh="${rsync_rsh} -i ${SSH_IDENTITY_FILE}"
fi

rsync_base=(
  rsync -az
  --update
  --partial
  --timeout="$RSYNC_TIMEOUT"
  --contimeout="$SSH_CONNECT_TIMEOUT"
  --rsh="$rsync_rsh"
  --itemize-changes
)

if [ "$DRY_RUN" = "1" ]; then
  rsync_base+=(-n)
fi

ensure_remote_repo() {
  local q_remote_root
  q_remote_root="$(printf '%q' "$REMOTE_REPO_ROOT")"
  if ! remote_exec "test -d ${q_remote_root}/.git"; then
    log "远程仓库不存在或不是 git 仓库：${REMOTE_REPO_ROOT}"
    return 1
  fi
}

ensure_remote_exclude() {
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi

  if [ "$REMOTE_KEEP_WORKDIRS" != "1" ]; then
    return 0
  fi

  local q_remote_root
  q_remote_root="$(printf '%q' "$REMOTE_REPO_ROOT")"

  remote_exec "set -euo pipefail
    cd ${q_remote_root}
    excl='.git/info/exclude'
    touch \"\$excl\"
    add_line(){ line=\"\$1\"; grep -Fxq \"\$line\" \"\$excl\" || echo \"\$line\" >> \"\$excl\"; }
    add_line 'work_dirs/'
    add_line '.ipynb_checkpoints/'
    add_line '__pycache__/'
    add_line '*.pyc'
  " >/dev/null
}

sync_work_dirs() {
  if [ "$SYNC_WORKDIRS" != "1" ]; then
    return 0
  fi

  local q_remote_root
  q_remote_root="$(printf '%q' "$REMOTE_REPO_ROOT")"
  if ! remote_exec "test -d ${q_remote_root}/work_dirs"; then
    log "远端不存在 work_dirs，跳过同步：${REMOTE_REPO_ROOT}/work_dirs"
    return 0
  fi

  mkdir -p "$REPO_ROOT/work_dirs"
  log "同步 work_dirs（远端 -> 本地，--update：只覆盖本地较旧文件）"
  "${rsync_base[@]}" "${remote_target}:${REMOTE_REPO_ROOT}/work_dirs/" "${REPO_ROOT}/work_dirs/"
}

sync_git_changed_files() {
  if [ "$SYNC_GIT_CHANGED" != "1" ]; then
    return 0
  fi

  local q_remote_root
  q_remote_root="$(printf '%q' "$REMOTE_REPO_ROOT")"

  log "同步远端 git 变更文件（diff + untracked）"
  remote_exec "set -euo pipefail
    cd ${q_remote_root}
    git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

    git fetch -q origin 2>/dev/null || true

    branch=\$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo master)
    upstream=\"origin/\$branch\"
    has_upstream=0
    git show-ref --verify --quiet \"refs/remotes/\$upstream\" && has_upstream=1 || true

    {
      if [ \"\$has_upstream\" = \"1\" ]; then
        git diff --name-only --diff-filter=ACMRTUXB \"\$upstream..HEAD\" 2>/dev/null || true
      fi
      git diff --name-only --diff-filter=ACMRTUXB --cached 2>/dev/null || true
      git diff --name-only --diff-filter=ACMRTUXB 2>/dev/null || true
      git ls-files --others --exclude-standard 2>/dev/null || true
    } | awk 'NF' | sort -u
  " | "${rsync_base[@]}" --files-from=- "${remote_target}:${REMOTE_REPO_ROOT}/" "${REPO_ROOT}/"
}

remote_cleanup() {
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi

  if [ "$REMOTE_RESET" != "1" ] && [ "$REMOTE_CLEAN_UNTRACKED" != "1" ]; then
    return 0
  fi

  local q_remote_root
  q_remote_root="$(printf '%q' "$REMOTE_REPO_ROOT")"

  log "清理远端 git 工作区（reset=${REMOTE_RESET}, clean_untracked=${REMOTE_CLEAN_UNTRACKED}）"
  remote_exec "set -euo pipefail
    cd ${q_remote_root}
    git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

    git fetch -q origin 2>/dev/null || true

    branch=\$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo master)
    upstream=\"origin/\$branch\"
    if [ '${REMOTE_RESET}' = '1' ]; then
      if git show-ref --verify --quiet \"refs/remotes/\$upstream\"; then
        git reset --hard \"\$upstream\"
      else
        git reset --hard
      fi
    fi

    if [ '${REMOTE_CLEAN_UNTRACKED}' = '1' ]; then
      git clean -fd
    fi
  " >/dev/null
}

sync_once() {
  if ! remote_exec "echo ok" >/dev/null 2>&1; then
    log "SSH 连接失败：${remote_target}:${REMOTE_PORT}（请先配置免密 SSH，或检查网络/端口）"
    return 1
  fi

  ensure_remote_repo || return 1
  ensure_remote_exclude || true

  sync_work_dirs || return 1
  sync_git_changed_files || return 1

  remote_cleanup || return 1
  return 0
}

while true; do
  if sync_once; then
    log "同步完成。"
  else
    log "同步失败，稍后重试。"
  fi

  if [ "$ONCE" = "1" ]; then
    break
  fi
  sleep "$INTERVAL"
done
