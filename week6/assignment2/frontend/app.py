from flask import Flask, request, jsonify, render_template
import boto3
import uuid
from dotenv import load_dotenv
import os
load_dotenv()
app = Flask(__name__)

# ---- CONFIG ----
REGION = "us-east-1"
AGENT_ID = "CXXBHRNH0P"
AGENT_ALIAS_ID = "4OEYBPT10V"

client = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN")
)

# ---- ROUTES ----

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")

    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    response = client.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId="test-session-1",
        inputText=user_input
    )

    final_text = ""

    for event in response.get("completion", []):
        if "chunk" in event:
            final_text += event["chunk"]["bytes"].decode("utf-8")

    return jsonify({"reply": final_text})


if __name__ == "__main__":
    app.run(debug=True)
