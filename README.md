# multi-llm-cbt-chatbot
A multi-agent mental health chatbot built with LangGraph, integrating LLaMA, Qwen, and Gemini to deliver Cognitive Behavioral Therapy (CBT)-informed conversations with built-in safety supervision.


⚠️ Disclaimer: This chatbot is a research and learning tool, not a replacement for professional mental health care. It was designed with explicit safety checks and is intended for educational purposes only.



🔗 Live Demo on Hugging Face

📊 Final Presentation + Demo Video

Overview

Single LLMs can struggle with therapy-style conversations, sometimes going off-track or giving unsafe advice. This project explores whether a CBT-informed multi-agent system with specialized roles and built-in supervision can do better.

The system routes each user message through three agents in sequence:

User Input
    │
    ▼
┌─────────────────────────────┐
│  Agent 1: Thought Analyst   │  Identifies automatic thoughts from user input
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Agent 2: CBT Specialist    │  Detects cognitive distortions and core beliefs
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Agent 3: Response + Safety │  Generates therapist-style response with safety supervision
└────────────┬────────────────┘
             │
             ▼
        Final Response

LLaMA, Qwen, and Gemini are distributed across the pipeline, chosen for their complementary strengths in instruction following, reasoning, and natural language generation.

Features


Multi-agent LangGraph pipeline with explicit state passing between nodes
CBT-grounded reasoning across automatic thoughts, cognitive distortions, and core beliefs
Safety supervision layer that checks outputs before they reach the user
Live Gradio interface deployed on Hugging Face Spaces
Multi-LLM integration across LLaMA, Qwen, and Gemini


Tech Stack

ComponentTechnologyAgent orchestrationLangGraphLLMsLLaMA, Qwen, GeminiFrontend / deploymentGradio, Hugging Face SpacesLanguagePython

Project Context

Built as part of the Cognitive Science Student Association (CSSA) AI Projects Program at UC San Diego. Presented to Cognitive Science faculty and students at the end of the quarter.

Limitations


Not a substitute for professional mental health support
Models may still produce imperfect or inconsistent responses
Safety supervision reduces but does not eliminate risk of harmful outputs


Authors

Developed by a team of Cognitive Science students at UC San Diego.
