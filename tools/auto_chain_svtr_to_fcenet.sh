#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用途：
  监听指定 SVTR 训练 log，检测到训练结束后，在 tmux 会话 fcenet 中自动启动 FCENet 训练（并可自动 attach/switch 到该会话）。

用法：
  bash tools/auto_chain_svtr_to_fcenet.sh [选项]

选项：
  --log <path>        SVTR 的 log 路径（默认：work_dirs/svtr_direct_fudan_scene/20260104_170633/20260104_170633.log）
  --epoch <int>       认为训练完成的 epoch（默认：30）
  --val-iters <int>   最后一个 val 的迭代数（默认：498）
  --session <name>    tmux 会话名（默认：fcenet）
  --window <name>     tmux 窗口名（默认：auto_fcenet_finetune）
  --no-attach         仅启动，不自动 attach/switch
  --dry-run           只打印将执行的 tmux/训练命令，不实际启动
  -h, --help          显示帮助
EOF
}

LOG_FILE="work_dirs/svtr_direct_fudan_scene/20260104_170633/20260104_170633.log"
END_EPOCH=30
VAL_ITERS=498
TMUX_SESSION="fcenet"
TMUX_WINDOW_BASE="auto_fcenet_finetune"
ATTACH=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log)
      LOG_FILE="${2:?缺少 --log 参数值}"
      shift 2
      ;;
    --epoch)
      END_EPOCH="${2:?缺少 --epoch 参数值}"
      shift 2
      ;;
    --val-iters)
      VAL_ITERS="${2:?缺少 --val-iters 参数值}"
      shift 2
      ;;
    --session)
      TMUX_SESSION="${2:?缺少 --session 参数值}"
      shift 2
      ;;
    --window)
      TMUX_WINDOW_BASE="${2:?缺少 --window 参数值}"
      shift 2
      ;;
    --no-attach)
      ATTACH=0
      shift 1
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数：$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

SENTINEL_FILE="${SENTINEL_FILE:-"$(dirname "$LOG_FILE")/.auto_chain_to_fcenet.started"}"
if [[ -f "$SENTINEL_FILE" ]]; then
  echo "已检测到哨兵文件，认为已触发过：$SENTINEL_FILE"
  exit 0
fi

DONE_RE="Epoch\\(val\\) \\[${END_EPOCH}\\]\\[[[:space:]]*${VAL_ITERS}/${VAL_ITERS}\\]"
echo "等待 SVTR 训练结束..."
echo "- log：$LOG_FILE"
echo "- 结束标记（正则）：$DONE_RE"

while [[ ! -f "$LOG_FILE" ]]; do
  echo "log 不存在，继续等待：$LOG_FILE"
  sleep 10
done

if grep -qE "$DONE_RE" "$LOG_FILE"; then
  echo "log 已包含结束标记，直接触发后续任务。"
else
  echo "开始监听 log 追加内容（tail -F）..."
  grep -m 1 -E "$DONE_RE" < <(tail -n 0 -F "$LOG_FILE") >/dev/null
  echo "检测到结束标记，开始触发后续任务。"
fi

mkdir -p "$(dirname "$SENTINEL_FILE")"
date -Iseconds >"$SENTINEL_FILE"

if ! command -v tmux >/dev/null 2>&1; then
  echo "未找到 tmux，请先安装 tmux 或手动启动训练。" >&2
  exit 1
fi

FCENET_CMD="CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh configs/textdet/fcenet/fcenet_r50dcnv2_fpn_1500e_art_rctw_rects_finetune.py 2 --resume --work-dir work_dirs/fcenet_r50dcnv2_fpn_finetune_art_rctw_rects --cfg-options train_cfg.val_interval=2 default_hooks.checkpoint.interval=2"
LAUNCH_CMD="/bin/bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate openmmlab && cd /root/lanyun-tmp/mmocr && ${FCENET_CMD}'"

if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "tmux 会话不存在，创建：$TMUX_SESSION"
  tmux new-session -d -s "$TMUX_SESSION"
fi

WINDOW_NAME="$TMUX_WINDOW_BASE"
if tmux list-windows -t "$TMUX_SESSION" -F '#{window_name}' | grep -qx "$WINDOW_NAME"; then
  WINDOW_NAME="${WINDOW_NAME}_$(date +%Y%m%d_%H%M%S)"
fi

echo "将启动 FCENet 训练："
echo "- tmux session：$TMUX_SESSION"
echo "- tmux window ：$WINDOW_NAME"
echo "- command     ：$FCENET_CMD"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] tmux new-window -t \"$TMUX_SESSION\" -n \"$WINDOW_NAME\" \"$LAUNCH_CMD\""
  exit 0
fi

tmux new-window -t "$TMUX_SESSION" -n "$WINDOW_NAME" "$LAUNCH_CMD"
tmux select-window -t "$TMUX_SESSION:$WINDOW_NAME" || true

if [[ "$ATTACH" -eq 1 ]]; then
  if [[ -n "${TMUX-}" ]]; then
    tmux switch-client -t "$TMUX_SESSION" || true
  elif [[ -t 0 && -t 1 ]]; then
    tmux attach -t "$TMUX_SESSION"
  else
    echo "当前非交互 TTY，已启动训练但跳过 tmux attach。"
  fi
fi

