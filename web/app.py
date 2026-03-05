#!/usr/bin/env python3
"""
Text-to-SQL Web Interface
Flask backend for the text2sql_demo pipeline.
"""

import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, Response, jsonify, render_template, request

from text2sql_demo.pipeline import (
    extract_part4_inputs,
    run_part1_part2,
    run_part1_part3,
)
from text2sql_demo.part3 import default_db_path
from text2sql_demo.part4 import stream_part4

app = Flask(__name__,
    template_folder='templates',
    static_folder='static'
)


def get_env_or_default(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _sse_event(event: str, data: object) -> str:
    payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


def _make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_safe(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif obj is None:
        return None
    else:
        return str(obj)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        query_text = data.get("query", "").strip()
        top_m = int(data.get("top_m", 10))
        execute = bool(data.get("execute", True))

        if not query_text:
            return jsonify({"error": "Query text is required"}), 400

        if not execute:
            result = run_part1_part2(query_text, top_m=top_m)
            return jsonify(_make_json_safe(result))

        def generate():
            try:
                part123_result = run_part1_part3(
                    query_text,
                    top_m=top_m,
                    db_path=str(default_db_path()),
                    query_timeout_seconds=2.0,
                )
                yield _sse_event("part123", _make_json_safe(part123_result))

                part4_inputs = extract_part4_inputs(part123_result, db_path=str(default_db_path()))
                for token in stream_part4(**part4_inputs):
                    yield _sse_event("token", {"t": token})

                yield _sse_event("done", {})

            except Exception as e:
                yield _sse_event("error", {"error": str(e)})

        resp = Response(generate(), mimetype="text/event-stream")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    except Exception as e:
        return jsonify({"error": f"Invalid request: {str(e)}"}), 400


if __name__ == "__main__":
    port = int(get_env_or_default("PORT", "5000"))
    debug = get_env_or_default("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
