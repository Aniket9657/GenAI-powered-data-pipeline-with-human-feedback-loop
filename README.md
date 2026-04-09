# 🎨 GenAI Human Creativity Data Engine

## 🚀 Overview

This project is a GenAI-powered system designed to collect, process, and structure human creativity data from image-based interactions.

Users are shown an image (e.g., paintings, nature scenes) and asked to:

* Describe what they see
* Share their thoughts
* Write creative content (e.g., poems, emotions, interpretations)

The system uses Generative AI to transform raw human responses into structured datasets for AI training and analysis.

---

## 🧠 Core Idea

We simulate a real-world scenario similar to platforms like Instagram:

* A dummy account posts creative images (e.g., paintings)
* Audience engages by commenting:

  * Descriptions
  * Emotions
  * Poems
* The system collects and processes this data

👉 Goal: Build a dataset capturing **human creativity, perception, and emotional interpretation**

---

## 🎯 Use Case (Validated)

Example workflow:

1. Post a painting 🎨
2. Ask users:

   * "What do you feel about this?"
   * "Write a short poem inspired by this image"
3. Collect responses:

   * Emotional interpretations
   * Creative writing
4. Process using GenAI:

   * Clean text
   * Extract themes and emotions
   * Score creativity and quality

👉 This aligns with real industry needs:

* Creative AI training
* Emotion-aware systems
* Content recommendation engines

---

## 🏗️ System Architecture

### 1. Input Layer

* Image upload (painting, artwork)
* Prompt to guide user responses

### 2. User Interaction Layer

* Users submit:

  * Descriptions
  * Poems
  * Emotions

### 3. Data Collection Layer

* Store raw responses in database (SQLite)

### 4. GenAI Processing Layer

Using LLMs (local or API):

* Clean and normalize text
* Extract:

  * objects
  * emotion
  * theme
* Score:

  * quality
  * creativity

### 5. Structured Dataset

Output format:

```json
{
  "cleaned_text": "...",
  "objects": [],
  "emotion": "",
  "theme": "",
  "creativity_score": 0
}
```

### 6. Feedback Loop

* High-quality responses improve dataset
* Enables future AI model training

---

## ⚙️ Tech Stack

* Python 3.10+
* Streamlit (UI)
* SQLite (Database)
* Requests (API calls)
* GenAI:

  * Ollama (local LLM)
    OR
  * Hugging Face (free API)

---

## 🔁 Data Pipeline

Image → User Response → GenAI Processing → Structured Data → Storage → (Future Training)

---

## 🧠 GenAI Role

GenAI is used to:

* Clean noisy human input
* Extract structured meaning
* Evaluate creativity and relevance
* Generate additional synthetic data

---

## 📊 Features

* Image-based interaction
* Human creativity capture
* AI-powered text processing
* Structured dataset generation
* Quality scoring system

---

## 🚀 Future Improvements

* Multi-language support
* Emotion classification models
* Creative scoring algorithms
* Integration with social media APIs
* Dataset export for ML training

---

## 💡 Why This Matters

Modern AI systems rely heavily on human feedback.

This project:

* Captures real human perception
* Structures unstructured creativity
* Bridges human intelligence with AI systems

---

## 🧪 How to Run

```bash
pip install streamlit requests
streamlit run app.py
```

---

## 🧠 Inspiration

Inspired by how companies like Meta and OpenAI use human feedback to improve AI systems.

---

## 📌 Status

MVP (Minimum Viable Product) — Built for rapid prototyping and experimentation.
