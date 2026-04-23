# AutoStream AI Agent 

## Overview

This project implements a conversational AI agent for a fictional SaaS platform called AutoStream. The agent is designed to simulate a real-world sales and support assistant. It can understand user intent, answer product-related questions using a retrieval-based approach, and capture leads through a structured multi-step interaction.

The system combines intent classification, a Retrieval-Augmented Generation (RAG) pipeline, and a controlled workflow using LangGraph. It is capable of maintaining conversation state across multiple turns and triggering backend actions when required.

---

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/7w1k/aiAgent.git

```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory and add your API key:

```
AICREDITS_API_KEY=your_api_key_here
```

Important: The examiner or user must provide their own valid API key. The project will not run without it.

### 5. Run the application

```bash
python agent_bot.py
```

---

## Architecture Explanation

This project uses LangGraph to build a structured and stateful conversational agent. LangGraph was chosen because it allows precise control over execution flow using nodes and edges, which is particularly useful for multi-step interactions such as lead capture. Unlike simple chatbot loops, this approach ensures predictable transitions between different stages of the conversation.

The system consists of three primary components: intent detection, a RAG pipeline, and a lead capture workflow. The intent detection module uses a hybrid approach combining rule-based checks and a language model to classify user input into greeting, product inquiry, or lead intent.

State is managed through a shared dictionary (`AgentState`) that persists across multiple turns. This enables the agent to remember user-provided information such as name, email, and platform while guiding them through a structured sequence of questions.

The RAG pipeline uses a vector database (Chroma) to store embeddings generated from a Markdown knowledge base. When a product-related query is received, relevant chunks are retrieved and passed to the language model to generate accurate and context-aware responses.

---

## WhatsApp Integration (Webhook Design)

To integrate this agent with WhatsApp, the system can be connected using the WhatsApp Business API through providers such as Twilio or Meta Cloud API.

A backend service (for example, using Flask or FastAPI) would be set up to handle incoming webhook requests. When a user sends a message on WhatsApp, it is forwarded to the webhook endpoint. The server extracts the message content and passes it to the AI agent.

User sessions can be maintained by using the sender’s phone number as a unique identifier, allowing the system to preserve conversation state across messages. The agent’s response is then sent back to the user via the WhatsApp API.

This architecture enables real-time, stateful conversations over WhatsApp using the same core logic implemented in this project.

---

## Features

* Intent-based routing of user queries
* Retrieval-Augmented Generation for accurate responses
* Multi-turn state management
* Structured lead capture workflow
* Local storage of leads in JSON format

---
