# 🦄 UB Chat

Fast, private AI chat — a single-file Flask backend that wraps **Groq**, **Google Gemini**, **Serper** and live data (weather & crypto) into one deployable web service. Designed for **Render's free tier** and **UptimeRobot** pings.

Your conversations never leave the browser (`localStorage` only). The backend is stateless.

---

## ✨ Features

| | |
|---|---|
| 💬 **Streaming AI chat** | Token-by-token replies over NDJSON |
| 🔀 **3 model tiers** | `UB Quick Flash 2.4` · `UB Flash 3.5` · `UB Flash 3.8` |
| 🧠 **Smart fallback** | Groq (primary) → Gemini (backup) → spare Groq key |
| 🔎 **Live web search** | Serper — results cited as sources |
| 🖼️ **Image search** | Real image grid with links |
| 🎨 **Image generation** | Gemini image models (per-device daily limit) |
| 🌤️ **Live weather** | Open-Meteo, by **city name** — no location permission |
| 🪙 **Crypto prices** | CoinGecko, INR + USD |
| 🎙️ **Voice input** | MediaRecorder → Groq Whisper |
| 📎 **File reading** | Images, PDF, ZIP, plain-text & code files |
| 🔒 **Privacy** | Chats stored in browser only |
| ⏱️ **Rate limit** | 5 requests / minute / IP |
| ❤️ **Health check** | `GET /health` for UptimeRobot |

---

## 🚀 Deploy on Render (free tier)

### 1. Prepare your repo

Files at the root:
