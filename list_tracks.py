#!/usr/bin/env python3
import argparse
import json
import socket
import sys
from typing import Dict


def send_command(host: str, port: int, payload: Dict, timeout: float = 2.0) -> Dict:
    message = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.sendall(message)
            sock.shutdown(socket.SHUT_WR)
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
    except Exception as exc:
        raise RuntimeError(f"failed to send command: {exc}") from exc
    if not data:
        raise RuntimeError("empty response from server")
    try:
        return json.loads(data.decode("utf-8").strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid response: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description="List active tracks from the realtime fusion server.")
    parser.add_argument("--host", default="127.0.0.1", help="command server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=18100, help="command server port (default: 18100)")
    parser.add_argument("--timeout", type=float, default=2.0, help="command response timeout (seconds)")
    args = parser.parse_args()

    payload = {"cmd": "list_tracks"}

    try:
        resp = send_command(args.host, args.port, payload, timeout=args.timeout)
    except RuntimeError as exc:
        print(f"[list_tracks] {exc}", file=sys.stderr)
        sys.exit(1)

    status = resp.get("status")
    if status != "ok":
        message = resp.get("message", "unknown error")
        print(f"[list_tracks] command failed: {message}", file=sys.stderr)
        sys.exit(1)

    tracks = resp.get("tracks", [])
    count = resp.get("count", len(tracks))
    print(f"[list_tracks] {count} track(s)")
    for t in tracks:
        tid = t.get("id")
        state = t.get("state")
        color = t.get("color")
        conf = t.get("color_confidence")
        cx = t.get("cx"); cy = t.get("cy")
        yaw = t.get("yaw")
        print(f"  - id={tid} state={state} color={color} conf={conf:.2f} pos=({cx:.2f}, {cy:.2f}) yaw={yaw:.1f}")


if __name__ == "__main__":
    main()
