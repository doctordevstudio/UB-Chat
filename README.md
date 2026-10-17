# 🦄 UB Chat

A single-file Python (Flask) AI chat backend with a modern web UI, live streaming, web search, image search, image generation, weather, crypto prices, voice input, file analysis, and **live code preview** for HTML, CSS, JS, SVG, and PHP — all from one `app.py`.

Designed for **Render's free tier** and **UptimeRobot** monitoring. Zero database. Zero user accounts. Chats live only in the browser's `localStorage`.

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **Streaming AI chat** | Token-by-token replies via NDJSON streaming |
| 🔀 **Multi-model tiers** | `UB Quick Flash 2.4` · `UB Flash 3.5` · `UB Flash 3.8` |
| 🧠 **Smart fallback chain** | Groq (primary) → Gemini (backup) → spare Groq key |
| 🔎 **Live web search** | Serper → sources shown below answers |
| 🖼️ **Image search** | Real image grid (thumbnail + link + title) |
| 🎨 **Image generation** | Gemini image models (per-device daily limit) |
| 🌤️ **Weather** | Open-Meteo, by city name — no location permission |
| 🪙 **Crypto prices** | CoinGecko, INR + USD |
| 🎙️ **Voice input** | MediaRecorder → Groq Whisper |
| 📎 **File reading** | Images, PDF, ZIP, plain-text & code files |
| 💅 **Beautiful code blocks** | Syntax highlighting via highlight.js (GitHub Dark theme) |
| ▶️ **Live code preview** | Run HTML / CSS / JS / SVG / PHP code in a sandboxed iframe |
| 📋 **One-click copy** | Copy raw code to clipboard |
| 📐 **Link cards grid** | Web sources rendered as 2-column favicon + title cards |
| 🔒 **Privacy-first** | Chats never leave the browser |
| ⏱️ **Rate limit** | 5 requests / minute / IP |
| ❤️ **Health check** | `GET /health` for UptimeRobot pings |
| 🌗 **Dark theme aware** | Auto-adapts to system preferences |

---

## 🚀 Deploy on Render (free tier)

### 1. Repo structure

Your GitHub repo must contain:
