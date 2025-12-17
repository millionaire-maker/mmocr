#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Basic settings
# -----------------------------
: "${MIHOMO_PORT:=7890}"
: "${MIHOMO_API:=9090}"
: "${MIHOMO_LOG:=/var/log/mihomo/mihomo.log}"
: "${MIHOMO_PID:=/var/run/mihomo.pid}"
: "${MIHOMO_BIN:=/usr/local/bin/mihomo}"
: "${MIHOMO_DIR:=/etc/mihomo}"

: "${CODEX_SMOKE:=1}"
: "${CODEX_SKIP_GIT:=1}"
: "${SHOW_LOG_TAIL:=1}"
: "${LOG_TAIL_LINES:=80}"

: "${SHOW_PROVIDER_INFO:=1}"    # 1=show proxy-provider url/path/proxy count
: "${WARN_KEEPALIVE:=1}"        # 1=warn if keep-alive settings missing

: "${AUTO_FIX:=1}"              # 1=try to start mihomo if down
: "${AUTO_FIX_WAIT_SEC:=6}"     # wait for ports after starting

# -----------------------------
# US node auto selection
# -----------------------------
: "${AUTO_SELECT_US_NODE:=1}"   # 1=force select a US node in SELECT group
: "${PREFER_US_NODE:="[Hy2]🇺🇸 美国 03 2X 4837"}"
: "${US_KEYWORDS:="美国|🇺🇸|US"}"
: "${DELAY_TEST_URL:="https://www.gstatic.com/generate_204"}"
: "${DELAY_TIMEOUT_MS:=5000}"
: "${US_ROTATE_MAX:=8}"         # when 403, try up to N US nodes

GREEN="\033[1;32m"; YELLOW="\033[1;33m"; RED="\033[1;31m"; NC="\033[0m"
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
have() { command -v "$1" >/dev/null 2>&1; }
hr() { echo "------------------------------------------------------------"; }

proxy="http://127.0.0.1:${MIHOMO_PORT}"
api_root="http://127.0.0.1:${MIHOMO_API}"

# -----------------------------
# Helpers
# -----------------------------
curl_head() {
  local url="$1"; shift || true
  curl -I --max-time 15 "$@" "$url" 2>/dev/null | head -n 8 || true
}

curl_code() {
  # prints http status code only
  local url="$1"; shift || true
  curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$@" "$url" 2>/dev/null || echo "000"
}

resolve_mihomo_path() {
  local rel="${1:-}"
  if [[ -z "$rel" ]]; then
    return 1
  fi
  if [[ "$rel" == /* ]]; then
    echo "$rel"
    return 0
  fi
  # mihomo resolves relative paths from its -d directory
  echo "$MIHOMO_DIR/${rel#./}"
}

count_proxies_in_provider_yaml() {
  # Counts entries under top-level `proxies:` until next top-level key.
  local f="${1:-}"
  [[ -f "$f" ]] || { echo "0"; return 0; }
  awk '
    BEGIN { in_section=0; c=0 }
    /^proxies:[[:space:]]*$/ { in_section=1; next }
    in_section && /^[^[:space:]]/ { in_section=0 }
    in_section && $0 ~ /^[[:space:]]*-[[:space:]]*\\{[[:space:]]*name:/ { c++ }
    END { print c+0 }
  ' "$f" 2>/dev/null || echo "0"
}

list_proxy_providers_from_config() {
  local cfg="${1:-}"
  [[ -f "$cfg" ]] || return 0
  awk '
    BEGIN { in=0; name=""; url=""; path="" }
    /^proxy-providers:[[:space:]]*$/ { in=1; next }
    in && /^[^[:space:]]/ { in=0 }
    in && /^[[:space:]]{2}[^[:space:]]+:[[:space:]]*$/ {
      name=$1; sub(/:$/, "", name); url=""; path=""; next
    }
    in && /^[[:space:]]{4}url:[[:space:]]*/ {
      url=$0; sub(/^[[:space:]]{4}url:[[:space:]]*/, "", url)
      gsub(/^"/, "", url); gsub(/"$/, "", url)
      gsub(/^'\''/, "", url); gsub(/'\''$/, "", url)
      next
    }
    in && /^[[:space:]]{4}path:[[:space:]]*/ {
      path=$0; sub(/^[[:space:]]{4}path:[[:space:]]*/, "", path)
      gsub(/^"/, "", path); gsub(/"$/, "", path)
      gsub(/^'\''/, "", path); gsub(/'\''$/, "", path)
      if (name != "") print name "\t" url "\t" path
      next
    }
  ' "$cfg"
}

get_listener_pid() {
  ss -lntp 2>/dev/null \
    | awk -v p=":${MIHOMO_PORT}" '$4 ~ p {print $NF}' \
    | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
    | head -n1
}

ports_up() {
  ss -lnt 2>/dev/null | awk '{print $4}' | grep -qE "127\.0\.0\.1:${MIHOMO_PORT}$" \
  && ss -lnt 2>/dev/null | awk '{print $4}' | grep -qE "127\.0\.0\.1:${MIHOMO_API}$"
}

start_mihomo_nonsystemd() {
  mkdir -p "$(dirname "$MIHOMO_LOG")" "$(dirname "$MIHOMO_PID")"

  # Cleanup stale pidfile
  if [[ -f "$MIHOMO_PID" ]]; then
    old="$(cat "$MIHOMO_PID" 2>/dev/null || true)"
    if [[ -n "${old:-}" ]] && ! ps -p "$old" >/dev/null 2>&1; then
      rm -f "$MIHOMO_PID"
    fi
  fi

  if [[ ! -x "$MIHOMO_BIN" ]] && ! have mihomo; then
    fail "mihomo binary not found ($MIHOMO_BIN and PATH)."
    return 1
  fi

  local bin="$MIHOMO_BIN"
  if ! [[ -x "$bin" ]]; then
    bin="$(command -v mihomo)"
  fi

  if [[ ! -f "$MIHOMO_DIR/config.yaml" ]]; then
    fail "Config not found: $MIHOMO_DIR/config.yaml"
    return 1
  fi

  nohup "$bin" -d "$MIHOMO_DIR" >"$MIHOMO_LOG" 2>&1 &
  echo $! > "$MIHOMO_PID"
  ok "Started mihomo via nohup (pid=$(cat "$MIHOMO_PID"))"
}

try_autofix_start_mihomo() {
  [[ "$AUTO_FIX" != "1" ]] && return 0
  have ss || { warn "ss not found, cannot auto-fix ports"; return 0; }

  if ports_up; then
    return 0
  fi

  warn "mihomo ports not listening, trying AUTO_FIX start..."

  # systemd path if available
  if have systemctl && systemctl list-unit-files 2>/dev/null | grep -q '^mihomo\.service'; then
    systemctl start mihomo >/dev/null 2>&1 || true
  fi

  # fallback to nohup
  if ! ports_up; then
    start_mihomo_nonsystemd >/dev/null 2>&1 || true
  fi

  # wait for ports
  for _ in $(seq 1 "$AUTO_FIX_WAIT_SEC"); do
    if ports_up; then
      ok "mihomo ports are up after AUTO_FIX"
      return 0
    fi
    sleep 1
  done

  warn "AUTO_FIX attempted but ports still down."
  return 0
}

# -----------------------------
# US node selection helpers (Clash API)
# -----------------------------
uri_encode() {
  # requires jq
  printf '%s' "$1" | jq -sRr @uri
}

api_ok() {
  curl -s --max-time 3 "$api_root" >/dev/null 2>&1
}

get_select_json() {
  curl -s --max-time 8 "$api_root/proxies/SELECT" 2>/dev/null || true
}

list_select_nodes() {
  get_select_json | jq -r '.all[]?' 2>/dev/null || true
}

current_select_now() {
  get_select_json | jq -r '.now // empty' 2>/dev/null || true
}

set_select_node() {
  local node="$1"
  curl -s -X PUT "$api_root/proxies/SELECT" \
    -H 'Content-Type: application/json' \
    -d "$(jq -cn --arg n "$node" '{name:$n}')" >/dev/null 2>&1
}

get_delay_ms() {
  local node="$1"
  local enc; enc="$(uri_encode "$node")"
  curl -s --max-time 8 \
    "$api_root/proxies/${enc}/delay?timeout=${DELAY_TIMEOUT_MS}&url=$(uri_encode "$DELAY_TEST_URL")" \
  | jq -r '.delay // empty' 2>/dev/null
}

node_is_us() {
  local node="${1:-}"
  [[ -n "$node" ]] && echo "$node" | grep -Eq "$US_KEYWORDS"
}

ensure_us_select_node() {
  [[ "$AUTO_SELECT_US_NODE" != "1" ]] && return 0
  api_ok || return 0

  local all now
  all="$(list_select_nodes)"
  now="$(current_select_now)"

  [[ -z "$all" ]] && { warn "SELECT group .all is empty; cannot pick nodes"; return 0; }

  # If preferred exists, use it
  if echo "$all" | grep -Fxq "$PREFER_US_NODE"; then
    if [[ "$now" != "$PREFER_US_NODE" ]]; then
      if set_select_node "$PREFER_US_NODE"; then
        ok "Switched SELECT -> preferred US node: $PREFER_US_NODE"
      else
        warn "Failed to switch SELECT -> preferred node"
      fi
    else
      ok "SELECT already on preferred US node: $PREFER_US_NODE"
    fi
    return 0
  fi

  # If already US-ish, keep it
  if node_is_us "$now"; then
    ok "SELECT currently looks US-ish already: $now"
    return 0
  fi

  # Find US nodes
  local us_nodes
  us_nodes="$(echo "$all" | grep -E "$US_KEYWORDS" || true)"
  if [[ -z "$us_nodes" ]]; then
    warn "No US nodes matched ($US_KEYWORDS) in SELECT group; keeping current: ${now:-N/A}"
    return 0
  fi

  # Rank by delay, pick best
  local tmp="/tmp/mihomo_us_rank.$$"
  : > "$tmp"
  while IFS= read -r n; do
    [[ -z "$n" ]] && continue
    d="$(get_delay_ms "$n" || true)"
    if [[ "$d" =~ ^[0-9]+$ ]] && (( d > 0 )); then
      echo "${d}|${n}" >> "$tmp"
    fi
  done <<< "$us_nodes"

  local best=""
  if [[ -s "$tmp" ]]; then
    best="$(sort -n "$tmp" | head -n1 | cut -d'|' -f2-)"
  else
    best="$(echo "$us_nodes" | head -n1)"
  fi
  rm -f "$tmp" || true

  if [[ -n "$best" ]]; then
    if set_select_node "$best"; then
      ok "Switched SELECT -> US node: $best"
    else
      warn "Failed to switch SELECT -> US node"
    fi
  fi
}

openai_code_via_proxy() {
  # Without API key, expected 401. If 403 -> CF blocked.
  curl_code "https://api.openai.com/v1/models" --proxy "$proxy"
}

chatgpt_code_via_proxy() {
  curl_code "https://chatgpt.com/" --proxy "$proxy"
}

rotate_us_nodes_if_cf_blocked() {
  [[ "$AUTO_SELECT_US_NODE" != "1" ]] && return 0
  api_ok || return 0

  local code now all us_nodes
  code="$(openai_code_via_proxy)"
  [[ "$code" != "403" ]] && return 0

  warn "OpenAI returned 403 via proxy (likely Cloudflare block). Trying to rotate US nodes..."

  all="$(list_select_nodes)"
  us_nodes="$(echo "$all" | grep -E "$US_KEYWORDS" || true)"
  if [[ -z "$us_nodes" ]]; then
    warn "No US nodes available to rotate."
    return 0
  fi

  # Rank candidates by delay
  local tmp="/tmp/mihomo_us_rank_rotate.$$"
  : > "$tmp"
  while IFS= read -r n; do
    [[ -z "$n" ]] && continue
    d="$(get_delay_ms "$n" || true)"
    if [[ "$d" =~ ^[0-9]+$ ]] && (( d > 0 )); then
      echo "${d}|${n}" >> "$tmp"
    fi
  done <<< "$us_nodes"

  local candidates
  if [[ -s "$tmp" ]]; then
    candidates="$(sort -n "$tmp" | cut -d'|' -f2-)"
  else
    candidates="$us_nodes"
  fi
  rm -f "$tmp" || true

  now="$(current_select_now)"
  local tried=0
  while IFS= read -r n; do
    [[ -z "$n" ]] && continue
    [[ "$n" == "$now" ]] && continue

    ((tried+=1))
    if (( tried > US_ROTATE_MAX )); then
      break
    fi

    warn "Trying US node ($tried/$US_ROTATE_MAX): $n"
    set_select_node "$n" || continue
    sleep 1

    code="$(openai_code_via_proxy)"
    if [[ "$code" != "403" && "$code" != "000" ]]; then
      ok "Rotation success: OpenAI status via proxy is $code (good). Current SELECT: $(current_select_now)"
      return 0
    fi
  done <<< "$candidates"

  warn "Rotation did not find a non-403 OpenAI path within $US_ROTATE_MAX candidates. Current SELECT: $(current_select_now)"
  return 0
}

# -----------------------------
# Main
# -----------------------------
hr
echo "Proxy & Codex Health Check"
date
hr

have curl && ok "curl found" || { fail "curl not found"; exit 1; }
have ss && ok "ss found" || warn "ss not found (install iproute2)"
have jq && ok "jq found" || { warn "jq not found (apt-get install -y jq)"; }

hr
echo "1) mihomo process & ports"

try_autofix_start_mihomo

# refresh pidfile from listener if possible
if have ss; then
  live_pid="$(get_listener_pid || true)"
  if [[ -n "${live_pid:-}" ]]; then
    mkdir -p "$(dirname "$MIHOMO_PID")" || true
    echo "$live_pid" > "$MIHOMO_PID"
    ok "Detected mihomo from listener: pid=$live_pid (pidfile refreshed: $MIHOMO_PID)"
  fi
fi

if [[ -f "$MIHOMO_PID" ]]; then
  pid="$(cat "$MIHOMO_PID" 2>/dev/null || true)"
  if [[ -n "${pid:-}" ]] && ps -p "$pid" >/dev/null 2>&1; then
    ok "mihomo running (pid=$pid from $MIHOMO_PID)"
  else
    warn "PID file exists but process not running (pid=${pid:-N/A})"
  fi
else
  warn "PID file not found: $MIHOMO_PID"
fi

if have ss; then
  ports="$(ss -lntp 2>/dev/null | egrep "(:${MIHOMO_PORT}|:${MIHOMO_API})" || true)"
  if [[ -n "$ports" ]]; then
    ok "Listening ports found:"
    echo "$ports"
  else
    fail "No listening on :$MIHOMO_PORT or :$MIHOMO_API (mihomo may not be running)"
  fi
fi

hr
echo "2) mihomo API health"
if api_ok; then
  ok "API reachable: $api_root"
else
  fail "API NOT reachable: $api_root"
fi

if have jq; then
  sel="$(curl -s --max-time 8 "$api_root/proxies/SELECT" || true)"
  auto="$(curl -s --max-time 8 "$api_root/proxies/AUTO" || true)"
  if [[ -n "$sel" ]]; then echo "$sel" | jq '{name,type,now,all_length:(.all|length)}' && ok "SELECT group parsed" || true; else warn "SELECT group empty response"; fi
  if [[ -n "$auto" ]]; then echo "$auto" | jq '{name,type,now,all_length:(.all|length)}' && ok "AUTO group parsed" || true; else warn "AUTO group empty response"; fi
fi

# Force/ensure a US node, then rotate if CF blocks
ensure_us_select_node
rotate_us_nodes_if_cf_blocked

if [[ "$SHOW_PROVIDER_INFO" == "1" ]]; then
  hr
  echo "2.5) Proxy provider summary"
  cfg="$MIHOMO_DIR/config.yaml"
  if [[ -f "$cfg" ]]; then
    ok "Config: $cfg"
    while IFS=$'\t' read -r name url path; do
      [[ -z "${name:-}" ]] && continue
      echo "- provider: $name"
      [[ -n "${url:-}" ]] && echo "  url:  $url"
      [[ -n "${path:-}" ]] && echo "  path: $path"
      if [[ -n "${path:-}" ]]; then
        abs="$(resolve_mihomo_path "$path" || true)"
        if [[ -n "${abs:-}" && -f "$abs" ]]; then
          echo "  proxies_in_file: $(count_proxies_in_provider_yaml "$abs")"
        else
          warn "  provider file not found: ${abs:-$path}"
        fi
      fi
    done < <(list_proxy_providers_from_config "$cfg" || true)

    if [[ "$WARN_KEEPALIVE" == "1" ]]; then
      if ! grep -qE '^keep-alive-idle:' "$cfg"; then
        warn "keep-alive-idle not set in $cfg (streaming/SSE may drop on idle)."
      fi
      if ! grep -qE '^keep-alive-interval:' "$cfg"; then
        warn "keep-alive-interval not set in $cfg (streaming/SSE may drop on idle)."
      fi
    fi
  else
    warn "Config not found: $cfg"
  fi
fi

hr
echo "3) Proxy connectivity tests"
echo "$(curl_head "https://www.gstatic.com/generate_204" --proxy "$proxy")" | head -n 4 || true

openai_code="$(openai_code_via_proxy)"
echo "$(curl_head "https://api.openai.com/v1/models" --proxy "$proxy")" | head -n 8 || true
if [[ "$openai_code" == "401" ]]; then
  ok "OpenAI API reachable via proxy (401 expected without key)"
elif [[ "$openai_code" == "403" ]]; then
  warn "OpenAI returned 403 via proxy (Cloudflare block / bad exit IP). Try another US node."
else
  warn "OpenAI status via proxy is $openai_code (not clearly OK, but may still work)"
fi

chatgpt_code="$(chatgpt_code_via_proxy)"
if [[ "$chatgpt_code" == "200" || "$chatgpt_code" == "302" ]]; then
  ok "chatgpt.com reachable via proxy (status $chatgpt_code)"
elif [[ "$chatgpt_code" == "403" ]]; then
  warn "chatgpt.com returned 403 via proxy (Cloudflare block likely)"
else
  warn "chatgpt.com status via proxy is $chatgpt_code (uncertain)"
fi

hr
echo "4) Toolchain versions"
have node && ok "node: $(node -v)" || warn "node not found"
have npm  && ok "npm:  $(npm -v)" || warn "npm not found"
have codex && ok "codex: $(command -v codex)" || warn "codex not found"

hr
echo "5) Codex smoke test"
if [[ "$CODEX_SMOKE" == "1" ]] && have codex; then
  # Ensure env proxies are set for codex
  export HTTP_PROXY="$proxy" HTTPS_PROXY="$proxy"
  export ALL_PROXY="socks5h://127.0.0.1:${MIHOMO_PORT}"
  export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" all_proxy="$ALL_PROXY"

  args=()
  [[ "$CODEX_SKIP_GIT" == "1" ]] && args+=(--skip-git-repo-check)

  set +e
  out="$(codex exec "${args[@]}" "Say 'ok' and nothing else." 2>&1)"
  rc=$?
  set -e

  if [[ $rc -eq 0 ]] && echo "$out" | grep -qiE '(^ok$|[^a-z]ok[^a-z])'; then
    ok "Codex output looks OK"
    echo "$out" | tail -n 30
  else
    fail "Codex smoke test failed (rc=$rc)"
    echo "$out" | tail -n 120
    warn "Hints:"
    warn " - If you see 'Access blocked by Cloudflare (403)': switch SELECT to another US node"
    warn " - If you see 'Not inside a trusted directory': add --skip-git-repo-check or run git init"
  fi
else
  warn "Skip codex smoke test (CODEX_SMOKE=$CODEX_SMOKE or codex not found)"
fi

hr
echo "6) Logs"
if [[ "$SHOW_LOG_TAIL" == "1" ]]; then
  if [[ -f "$MIHOMO_LOG" ]]; then
    ok "Tail mihomo log ($MIHOMO_LOG):"
    tail -n "$LOG_TAIL_LINES" "$MIHOMO_LOG" || true
  else
    warn "mihomo log not found: $MIHOMO_LOG"
  fi
fi

hr
ok "Check completed."
