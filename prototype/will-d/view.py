import argparse, http.server, socketserver, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def build():
    subprocess.run([sys.executable, str(ROOT/"scripts/build_site.py"), "--results", str(ROOT/"results"), "--out", str(ROOT/"site")], check=True)
def serve(port):
    with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as h:
        print(f"serving {ROOT}/site at http://localhost:{port}")
        h.serve_forever()
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--build-only", action="store_true"); p.add_argument("port", nargs="?", type=int, default=8000); args=p.parse_args()
    build()
    if args.build_only: raise SystemExit
    import os; os.chdir(ROOT/"site")
    serve(args.port)
