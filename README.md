# 🌌 Skyra: A Smart AI-Powered Investment Manager

<p align="center">
  <img src="images/app.png" alt="Skyra App Interface" width="700"/>
</p>

**Skyra** is a next-generation multi-agent investment manager powered by GPT-4o and Retrieval-Augmented Generation (RAG). Designed to act as your autonomous portfolio team, Skyra ingests live market data, applies diverse analytical lenses, and recommends actionable steps for retail or institutional investors—**Buy**, **Hold**, or **Sell**—in real time.

---

## 🚀 Features

### 🔁 Real-Time Market Intelligence
- Live API integration (Twelve Data, Alpha Vantage, Financial Modeling Prep).
- Automated data transformation with Microsoft Fabric (Bronze → Silver → Gold).
- Supports real-time and batch ingestion.

### 🤖 Multi-Agent Reasoning (GPT-4o + Semantic Kernel)
Each agent specializes in a different investment strategy:

- **🧠 Warren Buffett Agent** – Value investing logic.
- **📊 Valuation Agent** – PE, DCF, Intrinsic value.
- **💬 Sentiment Agent** – Reddit, Twitter, news analysis.
- **📈 Technical Agent** – RSI, MACD, Bollinger Bands.
- **🧾 Fundamental Agent** – Balance sheet, EPS, cash flow.
- **⚠️ Risk Manager** – Risk scoring, volatility.
- **🧩 Portfolio Aggregator** – Compiles and ranks decisions.

**🖼️ Add image here:** `images/agent-flow.png`  
_> Diagram showing all agents and how data flows between them._

### 🧠 RAG-Powered Insights
- Uses **Azure Cognitive Search** + **LangChain**.
- Retrieves analyst notes, filings, and documents.
- Gives **context-aware decisions** backed by real documents.

**🖼️ Add image here:** `images/rag-diagram.png`  
_> RAG architecture: embedding, search index, grounding._

### 🔐 Secure Credential Management
- Credentials managed by **Azure Key Vault**
- No hard-coded secrets – uses Managed Identity & `.env` encryption
- Optional support for `pydantic-settings` for dev profiles

### 🏗️ Lakehouse Architecture with Microsoft Fabric
- Full Medallion architecture:  
  - Bronze: Raw API data  
  - Silver: Cleaned & transformed  
  - Gold: Final enriched datasets for agent use

**🖼️ Add image here:** `images/medallion.png`  
_> Fabric pipeline + Lakehouse ingestion pipeline._

### 📨 Reflex + Semantic Kernel Automation
- Autonomous Reflex trigger system powered by Semantic Kernel
- Generates AI-written summaries and email reports
- Delivery via SMTP (SendGrid or Outlook)

---

## 📊 Example Agent Output

```txt
Ticker: TSLA

📊 Valuation Agent: Slightly overvalued by 8%, based on DCF analysis.
💬 Sentiment Agent: Positive spike on Twitter and analyst upgrades.
📈 Technical Agent: RSI at 62, MACD bullish crossover.
⚠️ Risk Agent: Volatility increasing; watch position size.
🧩 Final Recommendation: HOLD (wait for entry point near support level).
