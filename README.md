
# 🔐 LLM Evaluation and Security Framework - https://llmevaluationandsecurity.streamlit.app/

The primary goal of this project is to evaluate the performance and security of large language models (LLMs). The evaluation framework is based on a two-part structure:

Reference Model:
This should be a high-tier, flagship model used as a benchmark and ground truth to assess other models. It acts as a judge in evaluating both performance and security of the target models.

Target Model:
This is the model under evaluation. The aim is to measure its performance and identify any security vulnerabilities.

For convenience, I used GPT-3.5 Turbo as the reference model, since it was the only flagship model readily available to me. I selected various target models based on availability and suitability. However, the framework is flexible — any flagship model can be used as a reference, and any LLM can be evaluated as a target, depending on your needs.

Important Notes:

When evaluating a specific LLM, ensure you have access to the necessary resources, such as API keys or runtime environments.

The reference model can be treated as a hyperparameter in this framework — you can swap it out depending on the use case.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen)](https://llmevaluationandsecurity.streamlit.app/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🚀 Project Overview

The **LLM Evaluation and Security Framework** is an open-source tool designed to **evaluate the performance and security of Large Language Models (LLMs)**. It compares any target model against a trusted **reference model** (default: `GPT-3.5 Turbo`) using 20+ well-defined performance and safety metrics, delivering detailed insights via a live web interface.

> ⚡ **Core Idea:** Treat the **reference model** as a "judge" and evaluate the **target model** for both *output quality* and *robustness/security*.

## 🧠 Objective

- ✅ Enable **performance benchmarking** of LLMs using semantic similarity, BLEU, ROUGE, contextual metrics, and answer relevancy.
- 🔒 Measure **security posture** using metrics like prompt injection resistance, privacy leakage, hallucination, bias, and toxicity.
- 🔁 Provide a **flexible evaluation framework** that supports:
  - Swappable reference models (e.g., GPT-3.5, GPT-4)
  - Plug-and-play target models (e.g., LLaMA, Gemma, Mixtral via Groq)
- 📊 Deliver a **dynamic visual report** and auto-generated analysis for each evaluation.
- 🌐 Deployable as a **Streamlit Cloud app** for zero-setup use.

## 🏗️ System Architecture

```
User Prompt
     ↓
Reference Model (GPT-3.5)         Target Model (via Groq)
     ↓                                     ↓
           →→→ Metric Evaluation Engine ←←←
                      ↓
              Streamlit Dashboard
```

## 🧪 Key Features

- **20+ Evaluation Metrics**:
  - **Performance**: Semantic Similarity, BLEU, ROUGE-L, Faithfulness, Contextual Recall/Precision
  - **Security**: Prompt Injection Detection, Privacy Leak Test, Hallucination Score, Bias Resistance, Toxicity & more
- **Automatic Report Generation** using GPT-3.5
- **Interactive Visualization** of scores via Streamlit
- **Multi-Model Support** via OpenAI-compatible APIs
- **Agentic AI Ready** (Future): Extendable to automate multi-model evaluation via orchestration

## 📈 Example Metrics

| Metric                  | Type        | Description                                      |
|-------------------------|-------------|--------------------------------------------------|
| Semantic Similarity     | Performance | Cosine similarity between ref and target outputs|
| BLEU Score              | Performance | n-gram precision overlap                         |
| Prompt Injection Score  | Security    | Resistance to injection prompts                 |
| Data Privacy Score      | Security    | Measures leakage of dummy PII                   |
| Response Consistency    | Security    | Stability across multiple runs                  |

## 🌍 Live Demo

Access the live app on Streamlit Cloud:  
👉 **[llmevaluationandsecurity.streamlit.app](https://llmevaluationandsecurity.streamlit.app/)**

## 🛠️ Tech Stack

- **Language**: Python 3.9+
- **Frontend**: Streamlit
- **Backend**: Groq API (target LLM), Nexus AI GPT-3.5 (reference LLM)
- **Libraries**:
  - `sentence-transformers`
  - `nltk`, `rouge`, `scikit-learn`
  - `plotly`, `matplotlib`, `dotenv`, `openai`, `groq`

## ⚙️ Setup Instructions

```bash
git clone https://github.com/yourusername/llm-eval-sec.git
cd llm-eval-sec

# Install dependencies
pip install -r requirements.txt

# Set API keys (Groq + Nexus)
touch .env
echo "GROQ_API_KEY=your_key_here" >> .env
echo "NEXUS_AI_API_KEY=your_key_here" >> .env

# Run the Streamlit app
streamlit run app.py
```

## 📌 Future Enhancements

- 🤖 Agentic AI: Automate model orchestration across APIs using LangChain or CrewAI
- 🌐 Hugging Face Integration: Evaluate any LLM locally or via inference endpoints
- 📚 Benchmark Suite: Automate prompt suite evaluations and scoring dashboards
- 🔐 Advanced Security: Add red-teaming, fact-checking, and adversarial test generators

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgments

- [Groq API](https://console.groq.com/)
- [NavigateLabs Nexus AI](https://nexus.navigatelabs.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [OWASP GenAI Security](https://owasp.org/www-project-generative-ai-security/)

**Author**: SIDDHARTH S  
📧 siddharth.insight@gmail.com
🌐 www.linkedin.com/in/siddharth-wayne
