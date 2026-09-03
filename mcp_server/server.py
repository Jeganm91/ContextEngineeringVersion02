"""
Custom MCP-style tool server for the Context Engineering RAG lab (Azure flavour).

NOTE ON SCOPE: for lab simplicity this exposes the tool over a plain HTTP/JSON
endpoint rather than the full MCP stdio/SSE transport, so it is easy to run
and restart on the VM alongside the Flask app. Conceptually it plays the same
role as an MCP tool: the AI application calls it, gets a tool result, and
folds that result into the model's context.

Tool: get_latest_pricing
Purpose: fetch current premium software pricing from a live vendor pricing
         feed that is external to the Azure AI Search index -- simulating a
         case where the freshest information (pricing changes monthly)
         lives outside the static knowledge base.
Input:   { "query": "<user's question>" }
Output:  { "status": "success", "content": "<clean, one-paragraph answer>",
           "source": "mcp-live-pricing" }
"""

import json
from flask import Flask, request, jsonify

app = Flask(__name__)

with open("pricing_data.json") as f:
    PRICING_DATA = json.load(f)


@app.route("/tools/get_latest_pricing", methods=["POST"])
def get_latest_pricing():
    _ = request.get_json(silent=True) or {}

    content = (
        f"{PRICING_DATA['catalog_name']} (version {PRICING_DATA['version']}, "
        f"effective {PRICING_DATA['effective_date']}): {PRICING_DATA['summary']} "
        f"Last reviewed by {PRICING_DATA['last_reviewed_by']}."
    )

    response = {
        "status": "success",
        "content": content,
        "source": "mcp-live-pricing",
    }
    return jsonify(response)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9001)
