import os
import keras
import keras_nlp
import gradio as gr
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tavily import TavilyClient

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the Keras NLP model from Hugging Face 
model_path = "rukayatadedeji/ddi-finetuned-gemma2"  
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(f"hf://{model_path}")

# Load the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_path = "rukayatadedeji/kaggle_ddi"
persist_directory = f"hf://{db_path}"
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 2, 'fetch_k': 50})

# Initialize web search client (ensure you add your Tavily API key in the environment variables)
api_key = os.getenv("TAVILY_API_KEY")
search_api = TavilyClient(api_key)

# Define useful functions
def retrieve_context_from_chroma(query, retriever):
    """Retrieve relevant context from Chroma database."""
    results = retriever.invoke(query)
    return "\n".join([result.page_content for result in results])
    
def search_web(query, search_api):
    """Fetch relevant web search results."""
    response = search_api.search(query, include_domains=['reference.medscape.com'])
    snippets = [result['content'] for result in response['results']]
    return "\n".join(snippets)
    
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
      
    # Step 2: Retrieve context from Chroma
    chroma_context = retrieve_context_from_chroma(message, retriever)
        
    # Step 3: Retrieve web search context
    web_context = search_web(message, search_api)
        
    # Step 4: Construct prompt with both Chroma and web search contexts
    prompt = self.get_full_prompt()
    full_prompt = f"{chroma_context}\n\n{web_context}\n\n{prompt}"
        
    # Generate response with full prompt
    response = self.model.generate(full_prompt, max_length=1024)
    result = response.replace(full_prompt, "")  # Extract only the new response
        
    # Add the result to chat history
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

with gr.Blocks(theme="compact") as demo:
    # Background color styling
    gr.Markdown("<style>body { background-color: #00FFFF; }</style>")
    
    # Header with headline and image
    with gr.Row():
        gr.Image(value="stock-photo-medicine-interaction-concept.jpg", label="", height=300, width=600)
        gr.Markdown("<h2 style='font-family:Arial; color:#333333;'>Gemma-powered Drug Interactions AI App</h2>")

    # Section divider
    gr.Markdown("---")  # Horizontal line divider
    
    # Chat interface with rounded corners for chatbox
    gr.Markdown("<style>.chat-box { border-radius: 10px; }</style>")
    gr.ChatInterface(
        chat_with_model, 
        type="messages", 
        description="Type your drug interactions question below!"
    )
    
    # Footer acknowledgment
    gr.Markdown(
        "<p style='text-align: center; color: #888888;'>"
        "Developed by Rukayat Adedeji as part of the 2024 Google KaggleX Fellowship.</p>"
    )
    
demo.launch(debug=True)
