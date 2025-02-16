import streamlit as st
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import asyncio
from dotenv import load_dotenv
import os
from PIL import Image
from bs4 import BeautifulSoup
import requests
import json
import re
import mimetypes  # For Content-Type checking

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

class FashionTrendAgent:
    def __init__(self, groq_api_key):
        self.llm = Groq(model="mixtral-8x7b-32768", api_key=groq_api_key)
        self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512
        self.parser = SimpleNodeParser.from_defaults()
        self.fashion_sites = [
            "https://www.whowhatwear.com/fashion-trends",
            "https://www.harpersbazaar.com/fashion/trends/",
            "https://www.vogue.com/fashion/trends",
            "https://www.elle.com/fashion/trend-reports/"
        ]
        self.initialize_index()

    def get_simplified_content(self):
        return [Document(text="""Current fashion trends include sustainable fashion, oversized silhouettes, bold colors, minimalist aesthetics, and vintage-inspired pieces. Popular colors this season are pastels, earth tones, and vibrant accents. Key pieces include oversized blazers, wide-leg pants, statement accessories, and sustainable materials.""")]

    def initialize_index(self):
        documents = self.get_simplified_content()
        self.index = self.create_fashion_index(documents)

    async def update_fashion_index(self):
        documents = []
        for url in self.fashion_sites:
            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")

                images = soup.find_all("img")
                for img in images:
                    src = img.get("src")
                    alt = img.get("alt")
                    if src:
                         try:
                            response = requests.head(src)  # Check the header, not the whole image
                            response.raise_for_status()   # Raise an exception for bad status codes
                            image_data = {"url": src, "alt": alt}  # URL is good, store it
                            documents.append(Document(text=json.dumps(image_data), metadata={"source": url}))
                         except requests.exceptions.RequestException:
                            st.warning(f"Skipping broken URL: {src}")  # Log or skip the broken URL
                         continue # Go to the next image
                    #documents.append(Document(text=json.dumps(image_data), metadata={"source": url}))

                docs = SimpleWebPageReader(html_to_text=True).load_data([url])
                documents.extend(docs)

            except requests.exceptions.RequestException as e:
                st.error(f"Error scraping {url}: {e}")
                continue

        if documents:
            self.index = self.create_fashion_index(documents)

    def create_fashion_index(self, documents):
        nodes = self.parser.get_nodes_from_documents(documents)
        return VectorStoreIndex(nodes)

    def get_response(self, query):
        try:
            if any(word in query.lower() for word in ['trend', 'latest', 'current', 'new', 'season']):
                query_engine = self.index.as_query_engine(response_mode=ResponseMode.TREE_SUMMARIZE)
                response = query_engine.query(query)
                return str(response)
            else:
                context = """You are a helpful fashion stylist with extensive knowledge of current trends, styling techniques, and fashion history. Provide practical, personalized fashion advice that is specific and actionable. When describing clothing or outfits, if you have access to image URLs, include them in your response as JSON objects with the keys "url" and "alt". For example: {"url": "image_url_here", "alt": "image description"}. Include as many relevant images as you can. Do not mention or reveal that the images are in JSON format."""

                prompt = f"""{context}\n\nUser Query: {query}\n\nPlease provide detailed fashion advice that includes:\n1. Specific recommendations tailored to the query\n2. Practical styling tips and combinations\n3. Color and pattern suggestions that work well together\n4. Accessory recommendations to complete the look\n5. Tips for adapting the advice to different body types and occasions\n\nFormat your response in a clear, conversational manner."""

                response = self.llm.complete(prompt)
                return response.text

        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("Please set your GROQ_API_KEY environment variable")
            st.stop()
        st.session_state.agent = FashionTrendAgent(groq_api_key)

async def update_fashion_index(agent): # modified to take agent as parameter
    await agent.update_fashion_index()

def display_images_from_response(response):
    image_json_strings = []  # Initialize OUTSIDE the try block

    try:
        image_json_strings = re.findall(r"\{.*?\}", response)
        for json_str in image_json_strings:
            try:
                image_data = json.loads(json_str)
                url = image_data.get("url")
                alt = image_data.get("alt")
                if url:
                    try:
                        response = requests.head(url, timeout=5) # Added timeout
                        response.raise_for_status()
                        content_type = response.headers.get('Content-Type')
                        if content_type and content_type.startswith('image/'):
                            st.image(url, caption=alt, use_container_width=True)
                        else:
                            st.warning(f"Invalid Content-Type for {url}: {content_type}")
                    except requests.exceptions.RequestException as e:
                        st.warning(f"Could not display image from {url}: {e}")
                    except Exception as e:
                        st.warning(f"Could not display image from {url}: {e}")  # Handle display errors
            except json.JSONDecodeError:
                st.warning("Invalid JSON in LLM response.")

    except Exception as e:
        st.error(f"Error displaying images: {e}")


def main():
    st.set_page_config(page_title="Haute-U AR Technologies Fashion Assistant", page_icon="👗", layout="centered")

    col1, col2 = st.columns([1, 2])

    with col1:
        try:
            image_path = os.path.join(os.path.dirname(__file__), "haute_u_logo.png")
            image = Image.open(image_path)
            st.image(image, width=150)
        except FileNotFoundError:
            st.error("Logo image not found. Please place 'haute_u_logo.png' in the same directory as the script.")
        except Exception as e:
            st.error(f"Error loading image: {e}")

    with col2:
        st.title("Haute-U AR Technologies Fashion Assistant")
        st.markdown("""Your personal fashion advisor powered by AI. Ask me about:
        - Current fashion trends
        - Styling advice
        - Outfit combinations
        - Color coordination
        - Accessory recommendations""")

    initialize_session_state()

    if st.button("🔄 Refresh Fashion Trends"):
        with st.spinner("Updating fashion content..."):
             asyncio.run(update_fashion_index(st.session_state.agent)) # call the async function

        st.success("Fashion content updated!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me about fashion..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.get_response(prompt)
                st.markdown(response)

                display_images_from_response(response)

                st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown("---")
    st.markdown("&copy; 2024 Haute-U AR Technologies. All rights reserved.")

if __name__ == "__main__":
    main()