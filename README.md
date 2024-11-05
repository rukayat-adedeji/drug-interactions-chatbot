# Gemma-powered Drug Interactions AI Chatbot

This repository contains a Gradio-based chatbot application fine-tuned on a conversational drug-interaction dataset using Google's Gemma model. The chatbot assists users with questions related to drug interactions, leveraging the Gemma Causal Language Model (Gemma CausalLM) fine-tuned with conversational-style data for better contextual responses.

## Features

- **Conversational drug interaction assistant:** This chatbot provides detailed responses about potential drug interactions based on user queries.
- **Turn-based interaction:** The app retains conversation history for a coherent and continuous dialogue.
- **Integrated with Gradio:** A user-friendly interface for interacting with the chatbot, deployable directly within a Hugging Face Space.

## Model

The application uses a fine-tuned version of Google's Gemma CausalLM model. This fine-tuned model (`rukayatadedeji/ddi-finetuned-gemma2`) is tailored for drug interaction-related questions, providing specific, contextual answers to user inquiries.

## How It Works

1. **Model Loading:** Loads the fine-tuned Gemma CausalLM model from Hugging Face.
2. **Conversation Management:** The `ChatState` class maintains conversation history, formatted according to the Gemma model's guidelines for turn-based dialogues.
3. **Response Generation:** When a user message is received, the model generates a response based on the conversation history and system prompts.
4. **User Interface:** A Gradio-based chat interface presents the interaction to users in a conversational format.

## Code Overview

- **`ChatState` Class:** Manages conversation state, adding user and model turns to the chat history.
- **`send_message` Method:** Handles message processing and generates model responses.
- **Gradio Chat Interface:** Uses Gradio's `ChatInterface` to display the conversation, with an option to share and debug.

## Usage

To run the app in a Hugging Face Space:

1. Clone this repository into a new Hugging Face Space.
2. Run the `app.py` file to launch the Gradio interface.
3. The app is accessible directly through the interface, where users can input questions and receive responses related to drug interactions.

## Requirements

- **Python Libraries:** The app requires `keras`, `keras_nlp`, and `gradio`.
- **Model Path:** Ensure the model path in the code (`rukayatadedeji/ddi-finetuned-gemma2`) matches the saved fine-tuned model on Hugging Face.
