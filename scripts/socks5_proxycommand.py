#!/root/miniconda/envs/openmmlab/bin/python
"""
SOCKS5 ProxyCommand helper for OpenSSH.

Usage:
  socks5_proxycommand.py <proxy_host> <proxy_port> <dest_host> <dest_port>

Example (ssh):
  ssh -o ProxyCommand="python3 socks5_proxycommand.py 127.0.0.1 7890 %h %p" user@host
"""

from __future__ import annotations

import os
import selectors
import socket
import struct
import sys
from typing import Optional


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    data = bytearray()
    while len(data) < nbytes:
        chunk = sock.recv(nbytes - len(data))
        if not chunk:
            raise ConnectionError("unexpected EOF from proxy")
        data.extend(chunk)
    return bytes(data)


def _socks5_connect(
    proxy_host: str,
    proxy_port: int,
    dest_host: str,
    dest_port: int,
    timeout: float,
) -> socket.socket:
    sock = socket.create_connection((proxy_host, proxy_port), timeout=timeout)
    sock.settimeout(timeout)

    # Greeting: ver=5, nmethods=1, method=0 (no auth)
    sock.sendall(b"\x05\x01\x00")
    resp = _recv_exact(sock, 2)
    if resp[:1] != b"\x05":
        raise ConnectionError(f"invalid SOCKS version in greeting response: {resp!r}")
    if resp[1] != 0x00:
        raise ConnectionError(f"proxy does not accept no-auth method, method={resp[1]}")

    dest_host_bytes = dest_host.encode("utf-8")
    if len(dest_host_bytes) > 255:
        raise ValueError("dest_host too long for SOCKS5 domain name")

    # CONNECT request with domain name
    req = b"\x05\x01\x00\x03" + bytes([len(dest_host_bytes)]) + dest_host_bytes + struct.pack("!H", dest_port)
    sock.sendall(req)

    # Reply: ver, rep, rsv, atyp, bnd.addr, bnd.port
    hdr = _recv_exact(sock, 4)
    ver, rep, _rsv, atyp = hdr[0], hdr[1], hdr[2], hdr[3]
    if ver != 0x05:
        raise ConnectionError(f"invalid SOCKS version in connect response: {ver}")
    if rep != 0x00:
        raise ConnectionError(f"proxy CONNECT failed, rep={rep}")

    if atyp == 0x01:  # IPv4
        _recv_exact(sock, 4)
    elif atyp == 0x03:  # domain
        ln = _recv_exact(sock, 1)[0]
        _recv_exact(sock, ln)
    elif atyp == 0x04:  # IPv6
        _recv_exact(sock, 16)
    else:
        raise ConnectionError(f"unknown ATYP in connect response: {atyp}")
    _recv_exact(sock, 2)  # bnd.port

    sock.settimeout(None)
    return sock


def _relay(sock: socket.socket) -> int:
    sel = selectors.DefaultSelector()
    sel.register(sock, selectors.EVENT_READ, data="sock")
    sel.register(sys.stdin, selectors.EVENT_READ, data="stdin")

    stdin_closed = False
    while True:
        for key, _mask in sel.select():
            if key.data == "stdin":
                chunk = os.read(sys.stdin.fileno(), 65536)
                if not chunk:
                    stdin_closed = True
                    try:
                        sock.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass
                    sel.unregister(sys.stdin)
                    continue
                sock.sendall(chunk)
            else:
                chunk = sock.recv(65536)
                if not chunk:
                    return 0
                try:
                    os.write(sys.stdout.fileno(), chunk)
                except BrokenPipeError:
                    return 0

        if stdin_closed:
            # Wait for peer close; loop continues until sock EOF.
            pass


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        sys.stderr.write(
            "Usage: socks5_proxycommand.py <proxy_host> <proxy_port> <dest_host> <dest_port>\n"
        )
        return 2

    proxy_host = argv[1]
    proxy_port = int(argv[2])
    dest_host = argv[3]
    dest_port = int(argv[4])

    timeout = float(os.environ.get("SOCKS5_TIMEOUT", "15"))

    sock: Optional[socket.socket] = None
    try:
        sock = _socks5_connect(proxy_host, proxy_port, dest_host, dest_port, timeout=timeout)
        return _relay(sock)
    except Exception as exc:
        sys.stderr.write(f"socks5_proxycommand error: {exc}\n")
        return 1
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
