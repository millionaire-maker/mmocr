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
# Routing group settings (IMPORTANT)
# -----------------------------
# Outer group: your rules route traffic to this group (often PROXY).
# If OUTER_GROUP exists and contains OUTER_TARGET, script can force it to point to OUTER_TARGET.
: "${OUTER_GROUP:=PROXY}"
: "${OUTER_TARGET:=SELECT}"
: "${FORCE_OUTER_TARGET:=1}"    # 1=force OUTER_GROUP -> OUTER_TARGET if possible

# Inner group: where actual nodes live (often SELECT with many nodes).
: "${INNER_GROUP:=SELECT}"

# -----------------------------
# US node auto selection
# -----------------------------
: "${AUTO_SELECT_US_NODE:=1}"   # 1=force select a US node in INNER_GROUP
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
  local url="$1"; shift || true
  curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$@" "$url" 2>/dev/null || echo "000"
}

api_get() {
  local path="$1"
  curl -s --max-time 8 "$api_root$path" 2>/dev/null || true
}

api_ok() {
  curl -s --max-time 3 "$api_root" >/dev/null 2>&1
}

group_json() {
  local g="$1"
  api_get "/proxies/$g"
}

group_exists() {
  local g="$1"
  group_json "$g" | jq -e '.name? != null' >/dev/null 2>&1
}

group_now() {
  local g="$1"
  group_json "$g" | jq -r '.now // empty' 2>/dev/null || true
}

group_all_list() {
  local g="$1"
  group_json "$g" | jq -r '.all[]?' 2>/dev/null || true
}

group_all_length() {
  local g="$1"
  group_json "$g" | jq -r '(.all|length) // 0' 2>/dev/null || echo "0"
}

group_has_member() {
  local g="$1" member="$2"
  group_all_list "$g" | grep -Fxq "$member"
}

set_group_now() {
  local g="$1" node="$2"
  curl -s -X PUT "$api_root/proxies/$g" \
    -H 'Content-Type: application/json' \
    -d "$(jq -cn --arg n "$node" '{name:$n}')" >/dev/null 2>&1
}

ensure_outer_points_to_target() {
  [[ "$FORCE_OUTER_TARGET" != "1" ]] && return 0
  api_ok || return 0

  if ! group_exists "$OUTER_GROUP"; then
    warn "Outer group not found: $OUTER_GROUP (skip forcing)"
    return 0
  fi
  if ! group_has_member "$OUTER_GROUP" "$OUTER_TARGET"; then
    warn "Outer group '$OUTER_GROUP' does not contain '$OUTER_TARGET' (skip forcing)"
    return 0
  fi

  local now
  now="$(group_now "$OUTER_GROUP")"
  if [[ "$now" == "$OUTER_TARGET" ]]; then
    ok "Outer routing already: $OUTER_GROUP -> $OUTER_TARGET"
    return 0
  fi

  if set_group_now "$OUTER_GROUP" "$OUTER_TARGET"; then
    ok "Forced outer routing: $OUTER_GROUP -> $OUTER_TARGET"
  else
    warn "Failed to force outer routing: $OUTER_GROUP -> $OUTER_TARGET"
  fi
}

resolve_mihomo_path() {
  local rel="${1:-}"
  [[ -n "$rel" ]] || return 1
  if [[ "$rel" == /* ]]; then
    echo "$rel"; return 0
  fi
  echo "$MIHOMO_DIR/${rel#./}"
}

count_proxies_in_provider_yaml() {
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
    BEGIN { in_section=0; name=""; url=""; path="" }
    /^proxy-providers:[[:space:]]*$/ { in_section=1; next }
    in_section && /^[^[:space:]]/ { in_section=0 }
    in_section && /^  [^[:space:]]+:[[:space:]]*$/ {
      name=$1; sub(/:$/, "", name); url=""; path=""; next
    }
    in_section && /^    url:[[:space:]]*/ {
      url=$0; sub(/^    url:[[:space:]]*/, "", url)
      gsub(/^"/, "", url); gsub(/"$/, "", url)
      gsub(/^'\''/, "", url); gsub(/'\''$/, "", url)
      next
    }
    in_section && /^    path:[[:space:]]*/ {
      path=$0; sub(/^    path:[[:space:]]*/, "", path)
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
  [[ -x "$bin" ]] || bin="$(command -v mihomo)"

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

  ports_up && return 0

  warn "mihomo ports not listening, trying AUTO_FIX start..."

  if have systemctl && systemctl list-unit-files 2>/dev/null | grep -q '^mihomo\.service'; then
    systemctl start mihomo >/dev/null 2>&1 || true
  fi

  ports_up || start_mihomo_nonsystemd >/dev/null 2>&1 || true

  for _ in $(seq 1 "$AUTO_FIX_WAIT_SEC"); do
    ports_up && { ok "mihomo ports are up after AUTO_FIX"; return 0; }
    sleep 1
  done

  warn "AUTO_FIX attempted but ports still down."
  return 0
}

# -----------------------------
# US node selection helpers (Clash API)
# -----------------------------
uri_encode() {
  printf '%s' "$1" | jq -sRr @uri
}

get_inner_json() {
  api_get "/proxies/$INNER_GROUP"
}

list_inner_nodes() {
  get_inner_json | jq -r '.all[]?' 2>/dev/null || true
}

current_inner_now() {
  get_inner_json | jq -r '.now // empty' 2>/dev/null || true
}

set_inner_node() {
  local node="$1"
  set_group_now "$INNER_GROUP" "$node"
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

ensure_us_inner_node() {
  [[ "$AUTO_SELECT_US_NODE" != "1" ]] && return 0
  api_ok || return 0

  local all now
  all="$(list_inner_nodes)"
  now="$(current_inner_now)"

  [[ -z "$all" ]] && { warn "$INNER_GROUP group .all is empty; cannot pick nodes"; return 0; }

  if echo "$all" | grep -Fxq "$PREFER_US_NODE"; then
    if [[ "$now" != "$PREFER_US_NODE" ]]; then
      if set_inner_node "$PREFER_US_NODE"; then
        ok "Switched $INNER_GROUP -> preferred US node: $PREFER_US_NODE"
      else
        warn "Failed to switch $INNER_GROUP -> preferred node"
      fi
    else
      ok "$INNER_GROUP already on preferred US node: $PREFER_US_NODE"
    fi
    return 0
  fi

  if node_is_us "$now"; then
    ok "$INNER_GROUP currently looks US-ish already: $now"
    return 0
  fi

  local us_nodes
  us_nodes="$(echo "$all" | grep -E "$US_KEYWORDS" || true)"
  if [[ -z "$us_nodes" ]]; then
    warn "No US nodes matched ($US_KEYWORDS) in $INNER_GROUP; keeping current: ${now:-N/A}"
    return 0
  fi

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
    if set_inner_node "$best"; then
      ok "Switched $INNER_GROUP -> US node: $best"
    else
      warn "Failed to switch $INNER_GROUP -> US node"
    fi
  fi
}

openai_code_via_proxy() {
  curl_code "https://api.openai.com/v1/models" --proxy "$proxy"
}

chatgpt_code_via_proxy() {
  curl_code "https://chatgpt.com/" --proxy "$proxy"
}

rotate_us_nodes_if_cf_blocked() {
  [[ "$AUTO_SELECT_US_NODE" != "1" ]] && return 0
  api_ok || return 0

  local code
  code="$(openai_code_via_proxy)"
  [[ "$code" != "403" ]] && return 0

  warn "OpenAI returned 403 via proxy (likely Cloudflare block). Trying to rotate US nodes in $INNER_GROUP ..."

  local all us_nodes
  all="$(list_inner_nodes)"
  us_nodes="$(echo "$all" | grep -E "$US_KEYWORDS" || true)"
  if [[ -z "$us_nodes" ]]; then
    warn "No US nodes available to rotate in $INNER_GROUP."
    return 0
  fi

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

  local now tried=0
  now="$(current_inner_now)"

  while IFS= read -r n; do
    [[ -z "$n" ]] && continue
    [[ "$n" == "$now" ]] && continue

    ((tried+=1))
    (( tried > US_ROTATE_MAX )) && break

    warn "Trying US node ($tried/$US_ROTATE_MAX): $n"
    set_inner_node "$n" || continue
    sleep 1

    code="$(openai_code_via_proxy)"
    if [[ "$code" != "403" && "$code" != "000" ]]; then
      ok "Rotation success: OpenAI status via proxy is $code (good). Current $INNER_GROUP: $(current_inner_now)"
      return 0
    fi
  done <<< "$candidates"

  warn "Rotation did not find a non-403 OpenAI path within $US_ROTATE_MAX candidates. Current $INNER_GROUP: $(current_inner_now)"
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
  for g in "$OUTER_GROUP" "$INNER_GROUP" AUTO GLOBAL; do
    if group_exists "$g"; then
      echo "$(group_json "$g")" | jq '{name,type,now,all_length:(.all|length)}' || true
      ok "$g group parsed"
    fi
  done
fi

# IMPORTANT: ensure real routing uses INNER_GROUP
ensure_outer_points_to_target

# Force/ensure a US node, then rotate if CF blocks (all on INNER_GROUP)
ensure_us_inner_node
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
      grep -qE '^keep-alive-idle:' "$cfg" || warn "keep-alive-idle not set in $cfg (streaming/SSE may drop on idle)."
      grep -qE '^keep-alive-interval:' "$cfg" || warn "keep-alive-interval not set in $cfg (streaming/SSE may drop on idle)."
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
  warn "OpenAI returned 403 via proxy (Cloudflare block / bad exit IP). Try another node in $INNER_GROUP."
else
  warn "OpenAI status via proxy is $openai_code (uncertain)"
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
    warn " - If you see 403 Cloudflare: ensure OUTER_GROUP ($OUTER_GROUP) points to $OUTER_TARGET, and rotate nodes in $INNER_GROUP"
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
