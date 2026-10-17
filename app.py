"""
UB Chat — single-file backend
Deploy on Render free tier with:
    gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 90

UptimeRobot ping endpoint:  GET /health
"""

import os
import io
import re
import json
import time
import base64
import zipfile
import threading
from collections import defaultdict
from datetime import datetime, timezone

import requests
from flask import Flask, request, jsonify, Response, send_from_directory

# ------------------------------------------------------------------
#  CONFIG
# ------------------------------------------------------------------
def _env(key, default=""):
    return os.environ.get(key, default).strip()

GROQ_KEYS = {
    "brain":  _env("GROQ_KEY_BRAIN"),
    "deep":   _env("GROQ_KEY_DEEP"),
    "speed":  _env("GROQ_KEY_SPEED"),
    "vision": _env("GROQ_KEY_VISION"),
    "audio":  _env("GROQ_KEY_AUDIO"),
    "spare":  _env("GROQ_KEY_SPARE"),
}

SEARCH_KEYS = [k.strip() for k in _env("SEARCH_KEYS").split(",") if k.strip()]
BACKUP_KEYS = [
    k.strip()
    for k in _env("BACKUP_KEYS", "").replace("\\n", "\n").splitlines()
    if k.strip()
]

GROQ_API   = "https://api.groq.com/openai/v1"
GEMINI_API = "https://generativelanguage.googleapis.com/v1beta/models"
SERPER_API = "https://google.serper.dev"

MODELS = {
    "UB Quick Flash 2.4": {
        "id": "openai/gpt-oss-20b", "role": "speed", "effort": "low",
        "tokens": 2048, "history": 6,
        "hint": "Reply as briefly as possible: a few short sentences, unless the user asks for detail.",
    },
    "UB Flash 3.5": {
        "id": "openai/gpt-oss-120b", "role": "brain", "effort": "low",
        "history": 10, "hint": "Give clear, helpful answers of medium length.",
    },
    "UB Flash 3.8": {
        "id": "openai/gpt-oss-120b", "role": "deep", "effort": "high",
        "history": 12, "hint": "Think carefully and give a complete, well-structured, detailed answer with reasons and examples.",
    },
}

VISION_MODEL  = _env("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
WHISPER_MODEL = _env("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
IMAGE_MODELS  = ["gemini-2.5-flash-image", "gemini-3.1-flash-image-preview"]

SYSTEM = (
    "You are UB Chat, a fast, accurate assistant from India. "
    "Answer directly and concisely unless asked for detail. "
    "Put code in fenced code blocks with the language name. "
    "Write in the user's language. Use a friendly, helpful tone. "
    "FORMAT RULES: never put code inside a table; never write HTML tags such as <br>; "
    "use a table only for short plain-text comparisons with at most 3 columns, otherwise use short bullet lists or numbered steps; "
    "keep paragraphs short. You cannot send files such as ZIP or PDF. If asked to, say so politely and offer to write the "
    "content, code or steps in the chat instead. If the user asks you to draw or make an image, tell them to write 'generate image of ...' in the message."
)
NAMEHINT = " If asked who made you or which model you are, say you are UB Chat (UBAI), the user's own AI assistant, and never name any underlying model or company."

RATE_LIMIT_PER_MIN = int(_env("RATE_LIMIT_PER_MIN", "5"))
RATE_WINDOW_SEC    = 60

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# ------------------------------------------------------------------
#  Rate limiting
# ------------------------------------------------------------------
_rate_lock  = threading.Lock()
_rate_state = defaultdict(list)

def client_ip():
    fwd = request.headers.get("X-Forwarded-For", "")
    return fwd.split(",")[0].strip() if fwd else (request.remote_addr or "unknown")

def rate_ok(ip):
    now = time.time()
    with _rate_lock:
        bucket = _rate_state[ip]
        while bucket and now - bucket[0] > RATE_WINDOW_SEC:
            bucket.pop(0)
        if len(bucket) >= RATE_LIMIT_PER_MIN:
            return False
        bucket.append(now)
        return True

def rate_guard(fn):
    def wrapper(*a, **k):
        if not rate_ok(client_ip()):
            return jsonify({"error": f"Rate limit: {RATE_LIMIT_PER_MIN} req/min."}), 429
        return fn(*a, **k)
    wrapper.__name__ = fn.__name__
    return wrapper

# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------
def err_text(raw):
    try:
        j = json.loads(raw)
        return (j.get("error") or {}).get("message") or j.get("message") or raw
    except Exception:
        return (raw or "Request failed")[:200]

def now_str():
    return datetime.now(timezone.utc).strftime("%A, %d %B %Y, %H:%M UTC")

# ------------------------------------------------------------------
#  AI Streaming
# ------------------------------------------------------------------
def groq_stream(model_name, messages, key):
    cfg = MODELS.get(model_name, MODELS["UB Flash 3.5"])
    payload = {"model": cfg["id"], "messages": messages, "stream": True}
    if cfg.get("effort"): payload["reasoning_effort"] = cfg["effort"]
    if cfg.get("tokens"): payload["max_completion_tokens"] = cfg["tokens"]

    r = requests.post(f"{GROQ_API}/chat/completions",
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                      json=payload, stream=True, timeout=30)
    if r.status_code != 200: raise RuntimeError(err_text(r.text))

    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"): continue
        chunk = line[5:].strip()
        if chunk == "[DONE]": break
        try:
            d = json.loads(chunk)
            delta = d["choices"][0]["delta"]
            if delta.get("content"): yield delta["content"]
        except Exception: continue

def try_groq(model_name, messages):
    cfg = MODELS.get(model_name, MODELS["UB Flash 3.5"])
    order = [cfg["role"], "brain", "speed", "deep"]
    seen = set()
    for role in order:
        k = GROQ_KEYS.get(role, "").strip()
        if not k or k in seen: continue
        seen.add(k)
        try:
            buf = ""
            for tok in groq_stream(model_name, messages, k):
                buf += tok
                yield tok
            if buf.strip(): return
        except Exception as e:
            print("groq fallback:", e)
            continue
    raise RuntimeError("Primary AI unavailable")

def gemini_stream(model_name, contents, key):
    cfg = MODELS.get(model_name, MODELS["UB Flash 3.5"])
    think = {"UB Quick Flash 2.4": "minimal", "UB Flash 3.5": "low", "UB Flash 3.8": "low"}.get(model_name, "low")
    sys = SYSTEM + NAMEHINT + " " + cfg["hint"] + " Current date and time: " + now_str() + "."
    body = {
        "systemInstruction": {"parts": [{"text": sys}]},
        "contents": contents,
        "generationConfig": {"thinkingConfig": {"thinkingLevel": think}},
    }
    r = requests.post(f"{GEMINI_API}/gemini-3.5-flash:streamGenerateContent?alt=sse",
                      headers={"Content-Type": "application/json", "x-goog-api-key": key},
                      json=body, stream=True, timeout=45)
    if r.status_code != 200: raise RuntimeError(err_text(r.text))
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"): continue
        try:
            j = json.loads(line[5:])
            parts = (j.get("candidates") or [{}])[0].get("content", {}).get("parts") or []
            for p in parts:
                if p.get("text") and not p.get("thought"): yield p["text"]
        except Exception: continue

# ------------------------------------------------------------------
#  Search
# ------------------------------------------------------------------
_search_cache = {}
_search_lock  = threading.Lock()

def serper_search(query, num=6):
    ck = query.lower() + "|" + str(num)
    with _search_lock:
        hit = _search_cache.get(ck)
        if hit and time.time() - hit[0] < 1800: return hit[1]
    if not SEARCH_KEYS: raise RuntimeError("Search not configured")
    last = "Search unavailable"
    for k in SEARCH_KEYS:
        try:
            r = requests.post(f"{SERPER_API}/search",
                              headers={"X-API-KEY": k, "Content-Type": "application/json"},
                              json={"q": query, "num": num}, timeout=12)
            if r.status_code == 200:
                data = r.json()
                with _search_lock: _search_cache[ck] = (time.time(), data)
                return data
            last = f"Search HTTP {r.status_code}"
        except Exception as e: last = str(e)
    raise RuntimeError(last)

def serper_images(query, num=9):
    if not SEARCH_KEYS: raise RuntimeError("Search not configured")
    last = "Search unavailable"
    for k in SEARCH_KEYS:
        try:
            r = requests.post(f"{SERPER_API}/images",
                              headers={"X-API-KEY": k, "Content-Type": "application/json"},
                              json={"q": query, "num": num}, timeout=12)
            if r.status_code == 200: return r.json()
            last = f"Images HTTP {r.status_code}"
        except Exception as e: last = str(e)
    raise RuntimeError(last)

def web_context(d):
    parts = []
    ab = d.get("answerBox")
    if ab: parts.append("Featured answer: " + (ab.get("answer") or ab.get("snippet") or ab.get("title") or ""))
    for i, o in enumerate((d.get("organic") or [])[:6], 1):
        parts.append(f"{i}. {o.get('title','')} — {o.get('snippet','')} ({o.get('link','')})")
    return ("\n\n[LIVE WEB RESULTS from Google. Answer the question above using them. Be concise.]\n" + "\n".join(parts))

# ------------------------------------------------------------------
#  Weather + Crypto
# ------------------------------------------------------------------
def weather(city):
    g = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"count": 1, "name": city}, timeout=10).json()
    p = (g.get("results") or [None])[0]
    if not p: raise RuntimeError("City not found: " + city)
    w = requests.get("https://api.open-meteo.com/v1/forecast",
                     params={"latitude": p["latitude"], "longitude": p["longitude"],
                             "current": "temperature_2m,wind_speed_10m,weather_code"}, timeout=10).json()["current"]
    return {"name": p["name"] + (", " + p["country"] if p.get("country") else ""),
            "t": w["temperature_2m"], "w": w["wind_speed_10m"], "s": _wx_text(w["weather_code"])}

def _wx_text(code):
    if code == 0: return "Clear sky"
    if code < 4: return "Partly cloudy"
    if code < 50: return "Fog"
    if code < 60: return "Drizzle"
    if code < 70: return "Rain"
    if code < 80: return "Snow"
    if code < 85: return "Rain showers"
    if code < 95: return "Snow showers"
    return "Thunderstorm"

COINS = {"bitcoin": ["Bitcoin", "btc"], "ethereum": ["Ethereum", "eth"], "solana": ["Solana", "sol"],
         "dogecoin": ["Dogecoin", "doge"], "ripple": ["XRP", "xrp"]}

def crypto_prices(ids):
    r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                     params={"vs_currencies": "inr,usd", "include_24hr_change": "true", "ids": ",".join(ids)},
                     timeout=12)
    if r.status_code != 200: raise RuntimeError("Price service busy")
    d = r.json()
    return [{"name": COINS[i][0], "inr": d[i]["inr"], "usd": d[i]["usd"], "chg": d[i].get("inr_24h_change")}
            for i in ids if i in d]

# ------------------------------------------------------------------
#  Image Gen
# ------------------------------------------------------------------
def make_image(prompt):
    if not BACKUP_KEYS: raise RuntimeError("Image making not set up")
    body = {"contents": [{"role": "user", "parts": [{"text": "Create an image: " + prompt}]}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}}
    last_err = "Image generation failed"
    start = int(time.time()) % len(BACKUP_KEYS)
    for mod in IMAGE_MODELS:
        for i in range(min(len(BACKUP_KEYS), 8)):
            k = BACKUP_KEYS[(start + i) % len(BACKUP_KEYS)]
            try:
                r = requests.post(f"{GEMINI_API}/{mod}:generateContent",
                                  headers={"Content-Type": "application/json", "x-goog-api-key": k},
                                  json=body, timeout=45)
                if r.status_code != 200:
                    last_err = err_text(r.text); continue
                j = r.json()
                parts = ((j.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                for p in parts:
                    d = p.get("inlineData") or p.get("inline_data")
                    if d:
                        return "data:" + (d.get("mimeType") or d.get("mime_type") or "image/png") + ";base64," + d["data"]
            except Exception as e: last_err = str(e)
    raise RuntimeError(last_err)

# ------------------------------------------------------------------
#  Router
# ------------------------------------------------------------------
RE_GENV = re.compile(r"\b(generate|create|make|draw|design|paint|render|imagine|banao|banado)\b", re.I)
RE_GENN = re.compile(r"\b(images?|pictures?|pics?|photos?|logo|poster|wallpaper|painting|illustration|artwork)\b", re.I)
RE_GENX = re.compile(r"\b(code|function|script|program|html|css|app|website|api|component|python|java|react|folder|zip|pdf)\b", re.I)
RE_IMG  = re.compile(r"\b(images?|photos?|pictures?|pics?|wallpapers?)\b", re.I)
RE_WX   = re.compile(r"\b(weather|temperature|mausam)\b", re.I)
RE_CX   = re.compile(r"\b(price|rate|worth|cost|value|how much|trading)\b", re.I)
RE_WEB  = re.compile(r"\b(latest|news|headlines|currently|right now|who won|release date|samachar|khabar|taaza)\b", re.I)

def route(text):
    if RE_GENV.search(text) and RE_GENN.search(text) and not RE_GENX.search(text):
        return {"k": "gen", "q": text}
    if RE_IMG.search(text) and not re.search(r"\b(describe|code|html|css|website)\b", text, re.I):
        m = re.search(r"\b(\d{1,2})\s+(?:\w+\s+)?(?:images?|photos?|pictures?|pics?)\b", text, re.I)
        return {"k": "img", "q": text, "n": min(int(m.group(1)), 20) if m else 9}
    if RE_WX.search(text):
        m = re.search(r"\b(?:in|of|for|at)\s+([^?.!,]+)", text, re.I)
        return {"k": "wx", "q": (m.group(1).strip() if m else "")}
    ids = [i for i in COINS if re.search(r"\b(" + i + r"|" + COINS[i][1] + r")\b", text, re.I)]
    if ids and RE_CX.search(text):
        return {"k": "cx", "ids": ids}
    if RE_WEB.search(text):
        return {"k": "web", "q": text}
    return {"k": "chat"}

# ------------------------------------------------------------------
#  Prompt Build
# ------------------------------------------------------------------
def build_messages(model_name, msgs, ctx="", attachment=None):
    cfg = MODELS.get(model_name, MODELS["UB Flash 3.5"])
    hist = [m for m in msgs if m.get("content") and not m.get("err")][-cfg["history"]:]
    sys = SYSTEM + NAMEHINT + " " + cfg["hint"] + " Current date and time: " + now_str() + "."
    out = [{"role": "system", "content": sys}]
    for m in hist:
        out.append({"role": "assistant" if m["role"] == "model" else "user", "content": m["content"]})
    if attachment and out: out[-1]["content"] += "\n\n" + attachment
    if ctx and out: out[-1]["content"] += ctx
    while len(out) > 1 and out[1]["role"] == "assistant": out.pop(1)
    return out

# ------------------------------------------------------------------
#  Chat Endpoint
# ------------------------------------------------------------------
def sse(payload): return json.dumps(payload) + "\n"

@app.route("/api/chat", methods=["POST"])
@rate_guard
def chat():
    data = request.get_json(silent=True) or {}
    model_name = data.get("model") or "UB Flash 3.5"
    history    = data.get("messages") or []
    text       = (data.get("message") or "").strip()
    attachment_text = data.get("attachment_text") or ""

    if not text and not attachment_text: return jsonify({"error": "Empty message"}), 400

    r = route(text or attachment_text)

    if r["k"] == "wx":
        try:
            d = weather(r["q"]) if r["q"] else None
            if not d: return jsonify({"error": "Ask like: weather in Delhi"}), 400
            return jsonify({"kind": "weather", "data": d,
                            "text": f"[Live weather] {d['name']}: {d['t']}°C, {d['s']}, wind {d['w']} km/h."})
        except Exception as e: return jsonify({"error": str(e)}), 502

    if r["k"] == "cx":
        try:
            prices = crypto_prices(r["ids"])
            return jsonify({"kind": "crypto", "data": prices,
                            "text": "[Live crypto] " + "; ".join(f"{p['name']}: ₹{p['inr']} / ${p['usd']}" for p in prices)})
        except Exception as e: return jsonify({"error": str(e)}), 502

    if r["k"] == "gen":
        try:
            url = make_image(r["q"])
            return jsonify({"kind": "image", "url": url, "text": f'Image made for: "{r["q"][:80]}"'})
        except Exception as e: return jsonify({"error": str(e)}), 502

    if r["k"] == "img":
        try:
            d = serper_images(r["q"], r["n"])
            images = [{"url": x.get("imageUrl") or x.get("link"),
                       "thumb": x.get("thumbnailUrl") or x.get("imageUrl"),
                       "title": x.get("title") or "image",
                       "link": x.get("link")} for x in (d.get("images") or [])[:r["n"]]]
            return jsonify({"kind": "images", "images": images, "text": f'[Image results for "{r["q"]}"]'})
        except Exception as e: return jsonify({"error": str(e)}), 502

    ctx, sources = "", []
    if r["k"] == "web":
        try:
            d = serper_search(r["q"])
            ctx = web_context(d)
            sources = [o.get("link") for o in (d.get("organic") or [])[:4]]
        except Exception: pass

    messages = build_messages(model_name,
                              history + [{"role": "user", "content": text or "Summarise this file."}],
                              ctx=ctx, attachment=attachment_text)

    def generate():
        yield sse({"type": "sources", "sources": sources})
        try:
            for tok in try_groq(model_name, messages):
                yield sse({"type": "delta", "text": tok})
            yield sse({"type": "done"})
            return
        except Exception as e: print("Primary failed:", e)

        if BACKUP_KEYS:
            contents = [{"role": "user" if m["role"] == "user" else "model",
                         "parts": [{"text": m["content"]}]} for m in messages[1:]]
            start = int(time.time()) % len(BACKUP_KEYS)
            for i in range(min(len(BACKUP_KEYS), 6)):
                try:
                    for tok in gemini_stream(model_name, contents, BACKUP_KEYS[(start + i) % len(BACKUP_KEYS)]):
                        yield sse({"type": "delta", "text": tok})
                    yield sse({"type": "done"})
                    return
                except Exception as e: print("Gemini fallback:", e)

        spare = GROQ_KEYS.get("spare", "")
        if spare:
            try:
                for tok in groq_stream(model_name, messages, spare):
                    yield sse({"type": "delta", "text": tok})
                yield sse({"type": "done"})
                return
            except Exception as e: print("Spare failed:", e)

        yield sse({"type": "error", "error": "Couldn't get a response right now. Please tap Retry."})

    return app.response_class(generate(), mimetype="application/x-ndjson")

# ------------------------------------------------------------------
#  Speech-to-text
# ------------------------------------------------------------------
@app.route("/api/transcribe", methods=["POST"])
@rate_guard
def transcribe():
    key = GROQ_KEYS.get("audio") or GROQ_KEYS.get("spare")
    if not key: return jsonify({"error": "Voice input not configured"}), 503
    f = request.files.get("file")
    if not f: return jsonify({"error": "No audio"}), 400
    lang = request.form.get("language", "en")[:2]
    files = {"file": (f.filename or "voice.webm", f.read(), f.mimetype or "audio/webm")}
    r = requests.post(f"{GROQ_API}/audio/transcriptions",
                      headers={"Authorization": f"Bearer {key}"},
                      files=files, data={"model": WHISPER_MODEL, "language": lang}, timeout=60)
    if r.status_code != 200: return jsonify({"error": err_text(r.text)}), 502
    return jsonify({"text": (r.json().get("text") or "").strip()})

# ------------------------------------------------------------------
#  File upload
# ------------------------------------------------------------------
TEXT_EXT = re.compile(
    r"\.(txt|md|json|csv|tsv|js|jsx|ts|tsx|py|html?|css|scss|java|kt|c|h|cpp|hpp|cs|go|rs|php|rb|sh|bat|ya?ml|xml|sql|swift|dart|lua|r|toml|ini|env|gradle|vue|svelte|log)$",
    re.I,
)

@app.route("/api/upload", methods=["POST"])
@rate_guard
def upload():
    f = request.files.get("file")
    if not f: return jsonify({"error": "No file"}), 400
    raw = f.read()
    if len(raw) > 12 * 1024 * 1024: return jsonify({"error": "File too large (max 12 MB)"}), 413
    name = f.filename or "file"
    mime = f.mimetype or ""
    try:
        if mime == "application/pdf" or name.lower().endswith(".pdf"):
            return jsonify({"name": name, "mime": "application/pdf",
                            "text": f"[PDF: {name}]", "base64": base64.b64encode(raw).decode()})
        if mime.startswith("image/"):
            return jsonify({"name": name, "mime": mime,
                            "text": f"[Image: {name}]", "base64": base64.b64encode(raw).decode()})
        if name.lower().endswith(".zip") or "zip" in mime:
            zf = zipfile.ZipFile(io.BytesIO(raw))
            skip = re.compile(r"(^|/)(node_modules|\.git|dist|build|__pycache__|\.idea|\.vscode)/", re.I)
            names = [n for n in zf.namelist() if not n.endswith("/") and not skip.search(n)]
            out = f"[ZIP: {name}, {len(names)} files]\nFile list:\n" + "\n".join("- " + n for n in names[:300]) + "\n"
            used = 0
            for n in names:
                if not TEXT_EXT.search(n) or used > 70000: continue
                try:
                    txt = zf.read(n).decode("utf-8", "ignore")[:9000]
                    out += f"\n--- {n} ---\n{txt}\n"
                    used += len(txt)
                except Exception: continue
            return jsonify({"name": name, "mime": "application/zip", "text": out})
        if TEXT_EXT.search(name) or mime.startswith("text/") or mime == "application/json":
            txt = raw.decode("utf-8", "ignore")[:60000]
            return jsonify({"name": name, "mime": "text/plain", "text": f"[File: {name}]\n{txt}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Unsupported file."}), 415

# ------------------------------------------------------------------
#  Frontend
# ------------------------------------------------------------------
HTML_LINES = [
'<!DOCTYPE html>',
'<html lang="en">',
'<head>',
'<meta charset="utf-8">',
'<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover,interactive-widget=resizes-content">',
'<meta name="theme-color" content="#3b6cf6">',
'<title>UB Chat</title>',
'<link rel="icon" type="image/png" href="/logo.png">',
'<link rel="apple-touch-icon" href="/logo.png">',
'<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">',
'<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>',
'<style>',
'*{box-sizing:border-box;margin:0}',
':root{color-scheme:light;--bg:#f5f6fa;--card:#fff;--tx:#141b34;--mu:#6b7280;--bd:#e6e8ef;--ac:#3b6cf6;--ac2:#ff6b00;--sh:0 2px 10px rgba(20,27,52,.06);--code-bg:#1e1e2e}',
':root[data-theme=dark]{color-scheme:dark;--bg:#0f1115;--card:#181b22;--tx:#e8eaf0;--mu:#8b93a3;--bd:#262a33;--ac:#6c8cff;--ac2:#ff8c1a;--sh:none;--code-bg:#0d1117}',
'html,body{height:100%}',
'body{background:var(--bg);color:var(--tx);font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;overflow:hidden;display:flex;height:100%}',
'button,select,textarea,input{font:inherit;color:inherit}',
'button{cursor:pointer;border:0;background:none}',
'#side{width:270px;background:var(--card);display:flex;flex-direction:column;padding:12px;gap:4px;border-right:1px solid var(--bd)}',
'.brand{display:flex;align-items:center;gap:10px;padding:6px 8px 12px;border-bottom:1px solid var(--bd);margin-bottom:6px}',
'.brand img{width:36px;height:36px;object-fit:contain}',
'.brand b{font-size:16px}',
'.new{padding:11px 14px;border-radius:14px;text-align:left;font-weight:700;background:var(--ac);color:#fff;margin-bottom:6px}',
'#list{flex:1;overflow:auto;display:flex;flex-direction:column;gap:2px}',
'.ci{display:flex;align-items:center;border-radius:10px;padding:1px 4px 1px 10px}',
'.ci.on,.ci:hover{background:var(--bg)}',
'.ci span{flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;padding:8px 0;cursor:pointer;font-size:15px}',
'.ci button{color:var(--mu);padding:6px 9px;font-size:18px}',
'#main{flex:1;display:flex;flex-direction:column;min-width:0}',
'#top{display:flex;align-items:center;gap:8px;padding:10px 12px;background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(10px);z-index:10}',
'#model{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:8px 12px;font-weight:700}',
'#msgs{flex:1;overflow:auto;padding:4px 14px 20px}',
'.w{max-width:780px;margin:0 auto}',
'.m{margin:16px 0;display:flex;gap:10px}',
'.m.u{justify-content:flex-end}',
'.m.u .b{background:linear-gradient(135deg,var(--ac),#6c8cff);color:#fff;padding:10px 16px;border-radius:20px 20px 6px 20px;max-width:85%;white-space:pre-wrap}',
'.av{width:34px;height:34px;border-radius:10px;flex:none;background:var(--card);border:1px solid var(--bd);display:flex;align-items:center;justify-content:center;overflow:hidden}',
'.av img{width:100%;height:100%;object-fit:contain;padding:2px}',
'.m.a .b{flex:1;min-width:0;overflow-wrap:anywhere}',
'.b code{background:var(--bg);padding:1px 6px;border-radius:6px;font:.88em ui-monospace,monospace}',
'.b a{color:var(--ac);text-decoration:none}',
'.b a:hover{text-decoration:underline}',
'.code{background:var(--code-bg);border-radius:14px;margin:12px 0;overflow:hidden;border:1px solid #333;box-shadow:0 4px 12px rgba(0,0,0,.2)}',
'.ch{display:flex;justify-content:space-between;align-items:center;padding:8px 14px;background:#2d2d3f;color:#aaa;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.5px}',
'.ch span{display:flex;align-items:center;gap:6px}',
'.ch .lang-dot{width:8px;height:8px;border-radius:50%;background:#f1fa8c}',
'.ch .btns{display:flex;gap:8px}',
'.ch button{color:#ccc;font-size:12px;padding:2px 8px;border-radius:6px;transition:background .15s}',
'.ch button:hover{background:#444;color:#fff}',
'.ch button.preview{color:#8be9fd}',
'.ch button.preview:hover{background:#8be9fd;color:#000}',
'.code pre{padding:14px 16px;overflow-x:auto;margin:0;font:13.5px/1.6 ui-monospace,Menlo,Consolas,monospace;color:#f8f8f2;background:var(--code-bg)}',
'.code pre code{background:none;padding:0;font:inherit;color:inherit}',
'.card{background:var(--card);border-radius:16px;padding:14px 16px;box-shadow:var(--sh);margin:8px 0}',
'.card .big{font-size:34px;font-weight:700}',
'.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px;margin:12px 0}',
'.gi{position:relative;border-radius:14px;overflow:hidden;background:var(--bd);display:block;color:inherit;text-decoration:none;box-shadow:0 2px 6px rgba(0,0,0,.1);transition:transform .15s}',
'.gi:hover{transform:scale(1.02)}',
'.gi img{width:100%;aspect-ratio:1;object-fit:cover;display:block}',
'.gi .ttl{display:block;padding:8px;background:var(--card);font-size:12px;line-height:1.3;max-height:44px;overflow:hidden}',
'.lgrid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}',
'.lcard{display:flex;align-items:center;gap:10px;padding:10px 12px;background:var(--card);border:1px solid var(--bd);border-radius:12px;text-decoration:none;color:inherit;min-width:0;transition:box-shadow .15s,transform .1s}',
'.lcard:hover{box-shadow:var(--sh);transform:translateY(-1px)}',
'.lcard img{width:20px;height:20px;flex:none;border-radius:4px;background:var(--bg)}',
'.lcard span{flex:1;min-width:0;font-size:13.5px;line-height:1.35;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}',
'.lcard small{display:block;color:var(--mu);font-size:11px;margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
'#bar{padding:4px 12px 12px}',
'#box{max-width:780px;margin:0 auto;display:flex;align-items:flex-end;gap:8px;background:var(--card);border-radius:26px;padding:8px;box-shadow:0 4px 18px rgba(20,27,52,.1);border:1px solid var(--bd)}',
'#in{flex:1;border:0;background:none;outline:0;resize:none;max-height:180px;padding:8px 0;line-height:1.5}',
'#go{width:40px;height:40px;border-radius:50%;background:linear-gradient(135deg,var(--ac),#6c8cff);color:#fff;font-size:19px;flex:none;display:flex;align-items:center;justify-content:center}',
'#go:active{transform:scale(.92)}',
'#empty{text-align:center;margin-top:6vh}',
'#empty img{width:130px;height:130px;object-fit:contain}',
'#empty h1{font-size:28px;font-weight:800}',
'#empty h1 span{background:linear-gradient(90deg,var(--ac2),var(--ac),#138808);-webkit-background-clip:text;background-clip:text;color:transparent}',
'#empty p{color:var(--mu);margin:4px 0 18px}',
'.chip{display:inline-block;background:var(--card);border:1px solid var(--bd);border-radius:20px;padding:6px 14px;font-size:14px;margin:4px 6px 4px 0}',
'#toast{position:fixed;left:50%;bottom:96px;transform:translateX(-50%);background:var(--tx);color:var(--bg);padding:9px 16px;border-radius:12px;font-size:14px;opacity:0;pointer-events:none;transition:opacity .2s;z-index:60}',
'#toast.on{opacity:1}',
'#preview{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;z-index:100;flex-direction:column}',
'#preview.on{display:flex}',
'#preview-bar{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;background:#1e1e2e;border-bottom:1px solid #333;color:#fff}',
'#preview-bar b{font-size:15px}',
'#preview-bar button{color:#fff;font-size:14px;padding:6px 12px;border-radius:8px;background:#333}',
'#preview-bar button:hover{background:#444}',
'#preview iframe{flex:1;border:0;background:#fff;width:100%;height:100%}',
'@media(max-width:800px){#side{display:none}.m.a .b{font-size:16px}}',
'@media(max-width:520px){.lgrid{grid-template-columns:1fr}}',
'</style>',
'</head>',
'<body>',
'<aside id="side">',
'  <div class="brand"><img src="/logo.png" alt="UB Chat"><b>UB Chat</b></div>',
'  <button class="new" id="new">+ New chat</button>',
'  <div id="list"></div>',
'</aside>',
'<section id="main">',
'  <header id="top">',
'    <select id="model"></select>',
'    <span style="flex:1;text-align:right;color:var(--mu);font-size:13px" id="clock"></span>',
'  </header>',
'  <div id="msgs"><div class="w" id="w"></div></div>',
'  <div id="bar">',
'    <div id="ab" style="max-width:780px;margin:0 auto"></div>',
'    <input type="file" id="fi" hidden accept="image/*,application/pdf,.pdf,.zip,text/*,.md,.json,.csv,.js,.ts,.py,.html,.css,.php">',
'    <div id="box">',
'      <button id="att" class="chip" style="padding:8px 10px;margin:0">📎</button>',
'      <textarea id="in" rows="1" placeholder="Ask UB Chat anything"></textarea>',
'      <button id="mic" class="chip" style="padding:8px 10px;margin:0">🎙️</button>',
'      <button id="go" aria-label="Send">↑</button>',
'    </div>',
'  </div>',
'</section>',
'<div id="toast"></div>',
'<div id="preview">',
'  <div id="preview-bar">',
'    <b>Live Preview</b>',
'    <div style="display:flex;gap:6px">',
'      <button id="pv-reload">Reload</button>',
'      <button id="pv-close" style="font-weight:700">Close ✕</button>',
'    </div>',
'  </div>',
'  <iframe id="pv-frame"></iframe>',
'</div>',
'<script>',
'const $=id=>document.getElementById(id);',
'const LS={g(k,d){try{const v=JSON.parse(localStorage.getItem(k));return v==null?d:v}catch(e){return d}},s(k,v){try{localStorage.setItem(k,JSON.stringify(v))}catch(e){}}};',
'const E=s=>String(s==null?"":s).replace(/[&<>\\x27"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",\'"\':"&quot;","\\x27":"&#39;"}[c]));',
'const uid=()=>Date.now().toString(36)+Math.random().toString(36).slice(2,6);',
'function toast(t){const el=$("toast");el.textContent=t;el.classList.add("on");clearTimeout(toast._t);toast._t=setTimeout(()=>el.classList.remove("on"),2500)}',
'let chats=LS.g("ub_chats",[]),cur=null,busy=false,ctl=null,pend=null,rec=null;',
'const MODELS=["UB Quick Flash 2.4","UB Flash 3.5","UB Flash 3.8"];',
'MODELS.forEach(n=>$("model").add(new Option(n,n)));',
'$("model").value=LS.g("ub_model",MODELS[1]);',
'$("model").onchange=()=>LS.s("ub_model",$("model").value);',
'const getChat=()=>chats.find(c=>c.id===cur);',
'const save=()=>LS.s("ub_chats",chats);',
'',
'/* ============ RAW code block storage ============ */',
'window.__codeBlocks=[];',
'',
'/* ============ Link grid ============ */',
'const URL_RE=/\\[([^\\]\\n]+)\\]\\((https?:\\/\\/[^)\\s]+)\\)|(https?:\\/\\/[^\\s<>)"\\x27]+)/g;',
'function linkGrid(raw){',
'  const seen=new Set();const items=[];let m;URL_RE.lastIndex=0;',
'  while((m=URL_RE.exec(raw))!==null){',
'    const title=(m[1]||"").trim();',
'    const url=(m[2]||m[3]||"").trim();',
'    if(!url||seen.has(url))continue;',
'    seen.add(url);',
'    let domain="";try{domain=new URL(url).hostname.replace(/^www\\./,"")}catch(e){}',
'    items.push({title:title||domain||url,url,domain});',
'  }',
'  if(!items.length)return "";',
'  const cards=items.map(it=>{',
'    const fav="https://www.google.com/s2/favicons?domain="+encodeURIComponent(it.domain)+"&sz=32";',
'    return \'<a class="lcard" href="\'+E(it.url)+\'" target="_blank" rel="noopener"><img src="\'+fav+\'" alt="" loading="lazy" onerror="this.style.display=\\\'none\\\'"><span>\'+E(it.title)+\'<small>\'+E(it.domain)+\'</small></span></a>\';',
'  }).join("");',
'  return \'<div class="lgrid">\'+cards+\'</div>\';',
'}',
'',
'/* ============ Markdown + code with RAW storage ============ */',
'function md(t){',
'  /* 1. Extract fenced code blocks from RAW text BEFORE escaping. */',
'  const blocks=[];',
'  const withPh=t.replace(/```(\\w*)\\n?([\\s\\S]*?)```/g,(m,lang,code)=>{',
'    const idx=blocks.length;',
'    const rawCode=code.replace(/\\n$/,"");',
'    blocks.push({lang:(lang||"code").trim(),code:rawCode});',
'    return "\\u0001CODEBLOCK"+idx+"\\u0001";',
'  });',
'  /* 2. Escape the remaining text (no code inside it now). */',
'  let html=E(withPh);',
'  /* 3. Apply markdown transforms. */',
'  html=html',
'    .replace(/\\[([^\\]\\n]+)\\]\\((https?:\\/\\/[^)\\s]+)\\)/g,(m,txt,url)=>\'<a href="\'+url+\'" target="_blank" rel="noopener">\'+txt+\'</a>\')',
'    .replace(/\\*\\*([^*\\n]+)\\*\\*/g,"<b>$1</b>")',
'    .replace(/^### (.*)$/gm,"<h4>$1</h4>")',
'    .replace(/^#{1,2} (.*)$/gm,"<h3>$1</h3>")',
'    .replace(/^(?:[-*] .*(?:\\n|$))+/gm,mm=>"<ul>"+mm.trim().split("\\n").map(x=>"<li>"+x.slice(2)+"</li>").join("")+"</ul>")',
'    .replace(/\\n{2,}/g,"<br><br>").replace(/\\n/g,"<br>");',
'  /* 4. Swap placeholders for rendered code blocks (raw stored globally). */',
'  html=html.replace(/\\u0001CODEBLOCK(\\d+)\\u0001/g,(m,idx)=>{',
'    const b=blocks[+idx];',
'    const lang=b.lang.toLowerCase();',
'    const canPreview=["html","php","css","js","javascript","svg"].includes(lang);',
'    const gi=window.__codeBlocks.length;',
'    window.__codeBlocks.push({lang:b.lang,code:b.code});',
'    return \'<div class="code"><div class="ch"><span><i class="lang-dot"></i> \'+E(b.lang)+\'</span><div class="btns">\'+(canPreview?\'<button class="preview" data-code-idx="\'+gi+\'">Preview ▶</button>\':"")+\'<button class="copy" data-code-idx="\'+gi+\'">Copy</button></div></div><pre><code class="language-\'+E(b.lang)+\'">\'+E(b.code)+\'</code></pre></div>\';',
'  });',
'  setTimeout(()=>{if(window.hljs)document.querySelectorAll(".code code").forEach(el=>hljs.highlightElement(el))},10);',
'  return html;',
'}',
'',
'/* ============ Message bubble ============ */',
'function bubble(m){',
'  if(m.role==="user")return \'<div class="m u"><div class="b">\'+E(m.content)+"</div></div>";',
'  let body;',
'  if(m.err){body=\'<span style="color:#e0453b">\'+E(m.content)+"</span>"}',
'  else{body=md(m.content||"")+linkGrid(m.content||"")}',
'  return \'<div class="m a"><i class="av"><img src="/logo.png" alt="UB"></i><div class="b">\'+body+"</div></div>";',
'}',
'',
'/* ============ Preview: use data: URL, NOT srcdoc ============ */',
'$("pv-close").onclick=()=>{$("preview").classList.remove("on");$("pv-frame").src="about:blank"};',
'$("pv-reload").onclick=()=>{const f=$("pv-frame");const s=f.src;f.src="about:blank";setTimeout(()=>{f.src=s},50)};',
'',
'function buildPreview(code,lang){',
'  lang=(lang||"").toLowerCase();',
'  const note=lang==="php"?\'<div style="background:#fff3cd;border:1px solid #ffc107;padding:10px 14px;border-radius:8px;margin:0 0 20px;font-size:13px;color:#856404;font-family:sans-serif">⚠️ PHP preview runs client-side: server-side code (databases, sessions, file I/O) will not execute. Only the HTML/CSS output is visible.</div>\':"";',
'  if(lang==="html"||lang==="svg"){',
'    return note+code;',
'  }',
'  if(lang==="css"){',
'    return \'<!DOCTYPE html><html><head><meta charset="utf-8"><style>\'+code+\'</style></head><body style="padding:24px;font-family:system-ui,sans-serif;max-width:800px;margin:0 auto">\'+note+\'<h1>Heading 1</h1><h2>Heading 2</h2><h3>Heading 3</h3><p>This is a paragraph with <strong>bold</strong>, <em>italic</em>, and <a href="#">link</a> text. Lorem ipsum dolor sit amet.</p><button>Button</button> <button disabled>Disabled</button><ul><li>Item one</li><li>Item two</li></ul><blockquote style="border-left:4px solid #ccc;padding-left:12px;color:#666">A quote block.</blockquote><hr><table border="1" cellpadding="6" style="border-collapse:collapse"><thead><tr><th>Col A</th><th>Col B</th></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody></table></body></html>\';',
'  }',
'  if(lang==="js"||lang==="javascript"){',
'    return \'<!DOCTYPE html><html><head><meta charset="utf-8"><style>body{font-family:system-ui,sans-serif;padding:24px;background:#fff}h2{margin-top:0}#console{background:#0d1117;color:#7ee787;padding:14px;border-radius:8px;font:13px/1.5 ui-monospace,monospace;white-space:pre-wrap;min-height:120px;overflow:auto;max-height:70vh}</style></head><body><h2>JS Console</h2><div id="console"></div><script>',
'      const out=document.getElementById("console");',
'      const fmt=v=>{try{return typeof v==="object"&&v!==null?JSON.stringify(v,null,2):String(v)}catch(e){return String(v)}};',
'      ["log","info","warn","error"].forEach(m=>{console[m]=(...a)=>{out.textContent+=a.map(fmt).join(" ")+"\\\\n"}});',
'      window.onerror=(m)=>{out.textContent+="ERROR: "+m+"\\\\n"};',
'      try{'+code+'}catch(e){out.textContent+="THROWN: "+e.message+"\\\\n"}',
'    </script></body></html>\';',
'  }',
'  if(lang==="php"){',
'    let preview=code;',
'    preview=preview.replace(/<\\?php\\s*(?:echo|print)\\s+(["\'])([\\s\\S]*?)\\1\\s*;?\\s*\\?>/g,"$2");',
'    preview=preview.replace(/<\\?php[\\s\\S]*?\\?>/g,"<!-- PHP block -->");',
'    preview=preview.replace(/<\\?[\\s\\S]*?\\?>/g,"");',
'    return \'<!DOCTYPE html><html><head><meta charset="utf-8"></head><body style="padding:24px;font-family:system-ui,sans-serif">\'+note+preview+"</body></html>";',
'  }',
'  return \'<!DOCTYPE html><html><body style="padding:24px"><pre>\'+E(code)+\'</pre></body></html>\';',
'}',
'',
'/* ============ Click handlers ============ */',
'document.addEventListener("click",e=>{',
'  const copyBtn=e.target.closest(".copy");',
'  if(copyBtn){',
'    const idx=+copyBtn.dataset.codeIdx;',
'    const b=window.__codeBlocks[idx];',
'    if(b){navigator.clipboard.writeText(b.code).then(()=>{copyBtn.textContent="Copied!";setTimeout(()=>copyBtn.textContent="Copy",1500)}).catch(()=>toast("Copy failed"))}',
'    return;',
'  }',
'  const pvBtn=e.target.closest(".preview");',
'  if(pvBtn){',
'    const idx=+pvBtn.dataset.codeIdx;',
'    const b=window.__codeBlocks[idx];',
'    if(!b)return;',
'    const html=buildPreview(b.code,b.lang);',
'    $("pv-frame").src="data:text/html;charset=utf-8,"+encodeURIComponent(html);',
'    $("preview").classList.add("on");',
'    return;',
'  }',
'});',
'',
'/* ============ Rest of the app ============ */',
'function render(){',
'  const c=getChat();',
'  if(!c||!c.msgs.length){',
'    $("w").innerHTML=\'<div id="empty"><img src="/logo.png" alt="UB Chat"><h1>Hello. <span>How can I help?</span></h1><p>Chats stay in your browser.</p>\'+',
'      ["What\'s the weather in London?","Photos of the Taj Mahal","Latest technology news","Explain recursion with code"].map(x=>\'<button class="chip ex">\'+E(x)+"</button>").join("")+"</div>";',
'  } else $("w").innerHTML=c.msgs.map(bubble).join("");',
'  $("msgs").scrollTop=$("msgs").scrollHeight;',
'  $("list").innerHTML=chats.map(c=>\'<div class="ci\'+(c.id===cur?" on":"")+\'" data-id="\'+c.id+\'"><span>\'+E(c.title)+\'</span><button class="del">×</button></div>\').join("");',
'}',
'function setGo(){go.textContent=busy?"■":"↑";go.style.background=busy?"#e0453b":""}',
'$("new").onclick=()=>{if(busy)return;cur=null;render();$("in").focus()};',
'$("list").onclick=e=>{const row=e.target.closest(".ci");if(!row)return;const id=row.dataset.id;',
'  if(e.target.classList.contains("del")){if(!confirm("Delete this chat?"))return;chats=chats.filter(c=>c.id!==id);if(cur===id)cur=null;save();render()}',
'  else if(!busy){cur=id;render()}};',
'$("w").onclick=e=>{if(e.target.classList.contains("ex"))send(e.target.textContent)};',
'$("go").onclick=()=>busy?ctl.abort():send();',
'const inp=$("in"),go=$("go");',
'inp.oninput=()=>{inp.style.height="auto";inp.style.height=Math.min(inp.scrollHeight,180)+"px"};',
'inp.onkeydown=e=>{if(e.key==="Enter"&&!e.shiftKey&&!matchMedia("(pointer:coarse)").matches){e.preventDefault();send()}};',
'$("att").onclick=()=>{if(!busy)$("fi").click()};',
'$("fi").onchange=async e=>{const f=e.target.files[0];e.target.value="";if(!f)return;',
'  toast("Reading file…");',
'  try{const fd=new FormData();fd.append("file",f);',
'    const r=await fetch("/api/upload",{method:"POST",body:fd});',
'    const j=await r.json();',
'    if(!r.ok)throw new Error(j.error||"Upload failed");',
'    pend=j;toast("Attached: "+j.name);$("ab").innerHTML=\'<span class="chip">📎 \'+E(j.name)+\' <b id="ax">✕</b></span>\';',
'    $("ax").onclick=()=>{pend=null;$("ab").innerHTML=""};',
'  }catch(er){toast(er.message)}',
'};',
'$("mic").onclick=async()=>{',
'  if(rec){rec.stop();return}',
'  if(!navigator.mediaDevices)return toast("Mic not available");',
'  try{',
'    const st=await navigator.mediaDevices.getUserMedia({audio:true});',
'    const mr=new MediaRecorder(st);const chunks=[];',
'    mr.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};',
'    mr.onstop=async()=>{st.getTracks().forEach(t=>t.stop());',
'      $("mic").textContent="🎙️";rec=null;',
'      const blob=new Blob(chunks,{type:mr.mimeType||"audio/webm"});',
'      if(blob.size<1500)return toast("Didn\'t catch that.");',
'      toast("Converting voice…");',
'      const fd=new FormData();fd.append("file",blob,"voice.webm");fd.append("language",(navigator.language||"en").slice(0,2));',
'      const r=await fetch("/api/transcribe",{method:"POST",body:fd});',
'      const j=await r.json();',
'      if(!r.ok)return toast(j.error||"STT failed");',
'      inp.value+=(inp.value?" ":"")+j.text;inp.focus();',
'    };',
'    mr.start();rec=mr;$("mic").textContent="⏹";toast("Listening… tap again to stop");',
'  }catch(e){toast("Mic permission needed")}',
'};',
'async function send(txt){',
'  let text=(txt||inp.value).trim();',
'  if((!text&&!pend)||busy)return;',
'  let c=getChat();',
'  if(!c){c={id:uid(),title:text.slice(0,40)||(pend&&pend.name)||"New chat",msgs:[]};chats.unshift(c);cur=c.id}',
'  const userMsg={role:"user",content:text||(pend?("Summarise this file: "+pend.name):"")};',
'  c.msgs.push(userMsg);',
'  const att=pend?{name:pend.name,text:pend.text,base64:pend.base64,mime:pend.mime}:null;',
'  if(pend){pend=null;$("ab").innerHTML=""}',
'  const m={role:"model",content:""};c.msgs.push(m);',
'  inp.value="";inp.style.height="auto";busy=true;setGo();render();',
'  ctl=new AbortController();',
'  try{',
'    const history=c.msgs.slice(0,-2).map(x=>({role:x.role,content:x.content}));',
'    const r=await fetch("/api/chat",{method:"POST",signal:ctl.signal,',
'      headers:{"Content-Type":"application/json"},',
'      body:JSON.stringify({',
'        model:$("model").value,',
'        message:userMsg.content,',
'        messages:history,',
'        attachment_text:att?(att.text+(att.base64?"\\n[Attached file: "+att.mime+"]":"")):""',
'      })});',
'    if(!r.ok){const j=await r.json().catch(()=>({}));throw new Error(j.error||("HTTP "+r.status))}',
'    const ct=r.headers.get("content-type")||"";',
'    if(ct.includes("application/json")){',
'      const j=await r.json();',
'      if(j.kind==="weather")m.content="🌤️ "+j.data.name+"\\n\\n**"+j.data.t+"°C** — "+j.data.s+"\\nWind "+j.data.w+" km/h";',
'      else if(j.kind==="crypto")m.content=j.data.map(p=>"**"+p.name+"** — ₹"+p.inr+" / $"+p.usd+(p.chg!=null?" ("+p.chg.toFixed(1)+"%)":"")).join("\\n");',
'      else if(j.kind==="image")m.content="![Made image]("+j.url+")\\n\\n"+j.text;',
'      else if(j.kind==="images")m.content=j.images.map(x=>"- ["+x.title+"]("+x.link+")").join("\\n");',
'      else m.content=j.text||"";',
'      render();busy=false;setGo();save();return;',
'    }',
'    const rd=r.body.getReader();const dec=new TextDecoder();let buf="";',
'    for(;;){',
'      const{done,value}=await rd.read();if(done)break;',
'      buf+=dec.decode(value,{stream:true});',
'      const lines=buf.split("\\n");buf=lines.pop();',
'      for(const ln of lines){',
'        if(!ln.trim())continue;',
'        try{const j=JSON.parse(ln);',
'          if(j.type==="delta"){m.content+=j.text;render()}',
'          else if(j.type==="error"){m.err=true;m.content=j.error}',
'        }catch(e){}',
'      }',
'    }',
'  }catch(e){if(e.name!=="AbortError"){m.err=true;m.content="Couldn\'t get a response. Try again."}}',
'  busy=false;setGo();save();render();',
'}',
'setInterval(()=>{$("clock").textContent=new Date().toLocaleString("en-GB",{weekday:"short",day:"numeric",month:"short",hour:"2-digit",minute:"2-digit"})},20000);',
'render();',
'</script>',
'</body>',
'</html>',
]

INDEX_HTML = "\n".join(HTML_LINES)

# ------------------------------------------------------------------
#  Routes
# ------------------------------------------------------------------
@app.route("/")
def index():
    return Response(
        INDEX_HTML,
        mimetype="text/html",
        headers={
            "Content-Type": "text/html; charset=utf-8",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store, max-age=0",
        },
    )

@app.route("/logo.png")
def logo():
    return send_from_directory(".", "logo.png", mimetype="image/png")

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "time": now_str(),
        "groq_keys": sum(1 for v in GROQ_KEYS.values() if v),
        "backup_keys": len(BACKUP_KEYS), "search_keys": len(SEARCH_KEYS),
        "rate_limit": f"{RATE_LIMIT_PER_MIN}/min per IP",
        "html_len": len(INDEX_HTML),
    })

@app.route("/favicon.ico")
def favicon():
    return ("", 204)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"UB Chat listening on :{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
