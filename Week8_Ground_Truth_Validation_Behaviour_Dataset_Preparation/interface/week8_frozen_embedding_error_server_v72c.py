
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import csv
import json
import mimetypes


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

STATIC = W8 / "interface" / "static_v72c"
PKG = W8 / "outputs" / "v72c_frozen_embedding_error_analysis_interface" / "Week8_Frozen_Embedding_Error_Analysis_Interface"

FILES = {
    "predictions": PKG / "week8_v72c_champion_predictions.csv",
    "errors": PKG / "week8_v72c_champion_errors.csv",
    "confusions": PKG / "week8_v72c_top_confusions.csv",
    "class_summary": PKG / "week8_v72c_per_class_error_summary.csv",
    "scan_summary": PKG / "week8_v72c_scanframe_error_summary.csv",
    "video_summary": PKG / "week8_v72c_video_error_summary.csv",
}


def load_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


DATA = {k: load_csv(v) for k, v in FILES.items()}


def under_root(path):
    try:
        Path(path).resolve().relative_to(ROOT.resolve())
        return True
    except Exception:
        return False


def summary():
    rows = DATA["predictions"]
    errors = [r for r in rows if str(r.get("correct", "")).lower() != "true"]

    by_policy = {}
    for r in rows:
        p = r.get("split_policy", "")
        by_policy.setdefault(p, {"total": 0, "correct": 0, "errors": 0})
        by_policy[p]["total"] += 1

        if str(r.get("correct", "")).lower() == "true":
            by_policy[p]["correct"] += 1
        else:
            by_policy[p]["errors"] += 1

    return {
        "prediction_rows": len(rows),
        "correct_rows": len(rows) - len(errors),
        "error_rows": len(errors),
        "by_policy": by_policy,
        "claim_boundary": "v72c is frozen embedding baseline error inspection only. It does not change GT or train a model.",
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def send_file(self, path):
        path = Path(path)

        if not path.exists() or not path.is_file():
            self.send_error(404, "File not found")
            return

        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        b = path.read_bytes()

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            return self.send_file(STATIC / "index.html")

        if path == "/api/summary":
            return self.send_json(summary())

        for key in DATA:
            if path == f"/api/{key}":
                return self.send_json(DATA[key])

        if path == "/image":
            qs = parse_qs(parsed.query)
            requested = unquote(qs.get("path", [""])[0])

            if not requested:
                self.send_error(400, "Missing path")
                return

            img = Path(requested)

            if not img.exists() or not under_root(img):
                self.send_error(403, "Image path not allowed")
                return

            return self.send_file(img)

        static_path = STATIC / path.lstrip("/")

        if static_path.exists():
            return self.send_file(static_path)

        self.send_error(404, "Not found")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8532)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)

    print(f"Week8 v72c frozen embedding error interface running at http://{args.host}:{args.port}/")
    print("Open locally via SSH tunnel / forwarded port: http://127.0.0.1:8532/")

    server.serve_forever()


if __name__ == "__main__":
    main()
