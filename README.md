# 🔬 AI-Powered Research Paper Analyzer

A multi-agent system that automatically reads academic research papers and generates a comprehensive research brief using **LangGraph** + **Google Gemini 1.5 Flash**.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Multi-Agent Orchestration** | LangGraph state machine with Boss Agent + 4 specialized sub-agents |
| **Quality Control Loop** | Review Agent scores each output (1-10); retries up to 2× if score < 7 |
| **Token-Efficient RAG** | Each agent receives only the relevant paper section (not the full text) — saves ~80% of tokens vs naive approach |
| **Free-Tier Friendly** | Uses Gemini 1.5 Flash with rate-limit back-off; ~18K tokens total per paper |
| **Streamlit UI** | Live agent pipeline progress, score badges, tabs per section, download button |

---

## 🏗️ Architecture

```
Input (PDF / URL / Text)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Boss Agent (Orchestrator)                              │
│  • Parses PDF → extracts sections → routes contexts     │
└─────────────────────┬───────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │  Paper Analyzer Agent  │  ← abstract, intro, methods, results
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │  Review Agent (QC)    │  score ≥ 7 → proceed; else retry (max 2×)
          └───────────┬───────────┘
                      │ approved
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
  Summary         Citations       Key Insights
  Agent           Agent           Agent
  (abstract +     (references     (results +
  conclusion)     section)        discussion)
       │              │              │
       └──────────────┼──────────────┘
                      ▼
             Review Agent (each)
                      │ all approved
                      ▼
          ┌─────────────────────┐
          │  Boss Agent         │
          │  Combines all       │
          │  → Research Brief   │
          └─────────────────────┘
```



---

## 🚀 Setup

### 1. Clone / download the project
```bash
git clone <your-repo>
cd research_analyzer
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Gemini API key
```bash
cp .env.example .env
# Edit .env and add:  GEMINI_API_KEY=your_key_here
```

Get a free key at: https://aistudio.google.com/app/apikey

### 5. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 📂 Project Structure

```
research_analyzer/
├── app.py                    # Streamlit UI
├── requirements.txt
├── .env.example
│
├── utils/
│   ├── pdf_processor.py      # PDF extraction + section parser + context builder
│   └── gemini_client.py      # Gemini API wrapper with retry / rate-limit handling
│
├── agents/
│   ├── analyzer.py           # Paper Analyzer Agent
│   ├── summarizer.py         # Summary Generator Agent
│   ├── citation_extractor.py # Citation Extractor Agent
│   ├── insights.py           # Key Insights Agent
│   └── reviewer.py           # Review Agent (QC)
│
└── graph/
    ├── state.py              # LangGraph state schema (TypedDict)
    └── workflow.py           # LangGraph graph: nodes, edges, conditionals
```


## 💡 Usage Tips

- **Best input**: arXiv PDF links (e.g. `https://arxiv.org/pdf/2301.00234`)
- **Rate limits**: Free tier = 15 requests/minute. The pipeline adds 2s delays between calls automatically.
- **Long papers**: Text is capped at 3,000 chars per section — the most relevant content is preserved.
- **Quality threshold**: If any agent scores < 7, it automatically regenerates (max 2 retries).

--



---

