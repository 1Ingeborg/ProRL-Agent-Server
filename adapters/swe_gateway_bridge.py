"""Local HTTP/Unix bridge. Container network remains none; only Polar is reachable."""
import argparse, http.client, http.server, os, socket, socketserver
from urllib.parse import urlsplit

class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=600)
        self.path = path
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *args):
        pass  # Never write request bodies or authorization into bridge logs.
    def do_GET(self):
        self.forward()
    def do_POST(self):
        self.forward()
    def forward(self):
        allowed = {"/v1/chat/completions", "/v1/models"}
        if urlsplit(self.path).path not in allowed:
            self.send_error(403, "Only Polar model routes are allowed")
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > 16 * 1024 * 1024:
            self.send_error(413)
            return
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in {"host", "connection", "content-length", "transfer-encoding"}}
        conn = (UnixHTTPConnection(self.server.unix_path) if self.server.mode == "tcp"
                else http.client.HTTPConnection("127.0.0.1", getattr(self.server, "upstream_port", 8100), timeout=600))
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse()
            # SWE-agent uses non-streaming chat completions. Buffer the response.
            data = response.read()
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
        except (OSError, http.client.HTTPException):
            self.send_error(502, "Polar gateway is unavailable")
        finally:
            conn.close()
            self.close_connection = True

class UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
class TCPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["unix", "tcp"], required=True)
    p.add_argument("--socket", required=True)
    p.add_argument("--port", type=int, default=18080)
    args = p.parse_args()
    if args.mode == "unix":
        if os.path.lexists(args.socket):
            raise SystemExit("Socket already exists; stop the owned bridge first")
        server = UnixServer(args.socket, Handler)
        os.chmod(args.socket, 0o600)
    else:
        server = TCPServer(("127.0.0.1", args.port), Handler)
    server.mode, server.unix_path = args.mode, args.socket
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if args.mode == "unix":
            os.unlink(args.socket)
if __name__ == "__main__":
    main()
