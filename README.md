# 🎨 AI Custom Design Studio (Deity Attire Agent)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harsh-00-007/deity-attire-agent/blob/main/deployment.ipynb)


An autonomous AI pipeline that acts as both a Design Researcher and a Vision Creator. 

This application takes a vague client request (e.g., "Krishna winter poshak"), uses an LLM to autonomously reason and construct an optimized search query, scrapes the web for high-resolution cultural reference images, and mathematically blends them together using a Stable Diffusion IP-Adapter to generate a completely novel, custom design.

## 🚀 System Architecture

This project is broken down into a dual-engine architecture:

1. **The Search Agent (LangChain + LLM + SerpApi)**
   - Utilizes `ChatGroq` (Llama 3.1) / `Google Gemini` for blazing-fast reasoning.
   - Pydantic structured outputs force the LLM to explain its reasoning and generate SEO-optimized image search strings.
   - Built-in graceful degradation (network timeout fallbacks) and anti-bot bypassing (filtering strict domains and injecting browser headers).

2. **The Vision Engine (PyTorch + Hugging Face Diffusers)**
   - Built on Stable Diffusion 1.5.
   - Implements an **IP-Adapter (Image Prompt Adapter)** to bypass traditional text-to-image limitations. It converts up to 4 selected reference images into mathematical vectors and blends their styles, colors, and zardosi embroidery patterns into a single, cohesive output.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Backend / Orchestration:** Python, LangChain, Requests
- **Machine Learning / Vision:** PyTorch, Hugging Face `diffusers`, IP-Adapter, CLIP Vision Encoder
- **APIs:** Groq (LLM), Google Gemini (LLM Backup), SerpApi (Google Images)
