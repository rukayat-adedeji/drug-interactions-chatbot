import os
import keras
import keras_nlp
import gradio as gr


# Load the Keras NLP model from Hugging Face 
model_path = "rukayatadedeji/ddi-finetuned-gemma2"  
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(f"hf://{model_path}")

# Define a Chat class that maintains conversation history
class ChatState():
  """
  Manages the conversation history for a turn-based chatbot
  Follows the turn-based conversation guidelines for the Gemma family of models
  documented at https://ai.google.dev/gemma/docs/formatting
  """

  __START_TURN_USER__ = "user\n"
  __START_TURN_MODEL__ = "model\n"
  __END_TURN__ = "\n"

  def __init__(self, model, system=""):
    """
    Initializes the chat state.

    Args:
        model: The language model to use for generating responses.
        system: (Optional) System instructions or bot description.
    """
    self.model = model
    self.system = system
    self.history = []

  def add_to_history_as_user(self, message):
      """
      Adds a user message to the history with start/end turn markers.
      """
      self.history.append(self.__START_TURN_USER__ + message + self.__END_TURN__)

  def add_to_history_as_model(self, message):
      """
      Adds a model response to the history with start/end turn markers.
      """
      self.history.append(self.__START_TURN_MODEL__ + message + self.__END_TURN__)

  def get_history(self):
      """
      Returns the entire chat history as a single string.
      """
      return "".join([*self.history])

  def get_full_prompt(self):
    """
    Builds the prompt for the language model, including history and system description.
    """
    prompt = self.get_history() + self.__START_TURN_MODEL__
    if len(self.system)>0:
      prompt = self.system + "\n" + prompt
    return prompt

  def send_message(self, message):
    """
    Handles sending a user message and getting a model response.

    Args:
        message: The user's message.

    Returns:
        The model's response.
    """
    self.add_to_history_as_user(message)
    prompt = self.get_full_prompt()
    response = self.model.generate(prompt, max_length=1024)
    result = response.replace(prompt, "")  # Extract only the new response
    self.add_to_history_as_model(result)
    return result


# Initialize the Chat object with the model
chat = ChatState(gemma_lm)


def chat_with_model(input, history):
    '''Generates a response from the finetuned Gemma model'''
    
    answer = chat.send_message(input)
    response = {"role": "assistant", "content": ""}
    response['content'] += answer
    yield response
    
# Create a simple gradio chat interface and launch it
demo = gr.ChatInterface(chat_with_model,
                        type="messages",
                        description = "Gemma-powered Drug Interactions AI App")

# Launch the demo
demo.launch(share=True, debug=True)
