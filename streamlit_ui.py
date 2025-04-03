from langgraph.types import Command
from typing import List, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import streamlit as st
import asyncio
import uuid
import json
import os
import pandas as pd

from agent_graph import farming_agent_graph


# Page configuration
st.set_page_config(
    page_title="Sustainable Farming Assistant 2.0",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling - minimal now that we're using Streamlit's chat components
st.markdown("""
<style>
    .stChatMessage {
        margin-bottom: 1rem;
    }
    .stChatMessage .content {
        padding: 0.5rem;
    }
    .stChatMessage .timestamp {
        font-size: 0.8rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

class UserContext(BaseModel):
    user_id: str
    farmer_advisor: List[Dict[str, Any]] = []
    market_researcher: List[Dict[str, Any]] = []

@st.cache_resource
def get_thread_id():
    return str(uuid.uuid4())

thread_id = get_thread_id()

# Initialize session state for chat history and user context
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "user_context" not in st.session_state:
    st.session_state.user_context = UserContext(
        user_id=str(uuid.uuid4()),
        farmer_advisor=[],
        market_researcher=[]
    )

if "processing_message" not in st.session_state:
    st.session_state.processing_message = None

# Function to handle user input
def handle_user_message(user_input: str):
    # Add user message to chat history immediately
    timestamp = datetime.now().strftime("%I:%M %p")
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    
    # Check if both files are uploaded
    all_details_given = bool(
        st.session_state.user_context.farmer_advisor and 
        st.session_state.user_context.market_researcher
    )
    if not all_details_given:
        # If files are missing, set a message to ask the user to upload them
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "Please upload both the Farmer Advisor and Market Researcher data files to proceed.",
            "timestamp": timestamp
        })
    else:
        # Set the message for processing in the next rerun
        st.session_state.processing_message = user_input

# Function to invoke the agent graph to interact with the Travel Planning Agent
async def invoke_agent_graph(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    initial_state = {
        "user_input": user_input,
        "all_details_given": bool(
        st.session_state.user_context.farmer_advisor and 
        st.session_state.user_context.market_researcher
        ),
        "farmer_advisor": st.session_state.user_context.farmer_advisor,
        "market_researcher": st.session_state.user_context.market_researcher
    }
    # First message from user
    if len(st.session_state.chat_history) == 1:
        async for msg in farming_agent_graph.astream(
                initial_state, config, stream_mode="custom"
            ):
                yield msg
    # Continue the conversation         
    else:
        async for msg in farming_agent_graph.astream(
            Command(resume=user_input), config, stream_mode="custom"         ### Hacky solution.. original Command(resume=user_input), config, stream.... (To add the chat to previous messages, now it doesn't work)
        ):
            yield msg

async def main():
    # Sidebar for user preferences
    with st.sidebar:
        st.title("Farming Preferences")
        
        st.subheader("About You")
        name = st.text_input("Your Name", value="Plantation name")
        
        # Initialize session state for file upload tracking
        if "farmer_advisor_uploaded" not in st.session_state:
            st.session_state.farmer_advisor_uploaded = False
        if "market_researcher_uploaded" not in st.session_state:
            st.session_state.market_researcher_uploaded = False

        st.subheader("Farmer Advisor Data")
        farmer_advisor_file = st.file_uploader("Upload Farmer Advisor CSV", type=["csv"])
        if farmer_advisor_file:
            farmer_advisor_data = pd.read_csv(farmer_advisor_file)
            farmer_advisor_data = farmer_advisor_data.to_dict(orient="records")
            st.session_state.user_context.farmer_advisor = farmer_advisor_data
            st.session_state.farmer_advisor_uploaded = True
            st.success("Farmer Advisor data uploaded!")
        else:
            if st.session_state.get("farmer_advisor_uploaded", False):
                st.session_state.user_context.farmer_advisor = []
                st.session_state.farmer_advisor_uploaded = False
                st.warning("Farmer Advisor data removed!")

        st.subheader("Market Researcher Data")
        market_researcher_file = st.file_uploader("Upload Market Researcher CSV", type=["csv"])
        if market_researcher_file:
            market_researcher_data = pd.read_csv(market_researcher_file)
            market_researcher_data = market_researcher_data.to_dict(orient="records")
            st.session_state.user_context.market_researcher = market_researcher_data
            st.session_state.market_researcher_uploaded = True
            st.success("Market Researcher data uploaded!")
        else:
            if st.session_state.get("market_researcher_uploaded", False):
                st.session_state.user_context.market_researcher = []
                st.session_state.market_researcher_uploaded = False
                st.warning("Market Researcher data removed!")
        
        if st.button("Save Preferences"):
            st.success("Preferences saved!")
        
        st.divider()
        
        if st.button("Start New Conversation"):
            st.session_state.chat_history = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.success("New conversation started!")

    # Main chat interface
    st.title("🌾 Sustainable Farming Assistant 2.0")
    st.caption("Give me the details for your farming needs and let me plan it for you!")

    # Display chat messages
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user", avatar=f"https://api.dicebear.com/7.x/avataaars/svg?seed={st.session_state.user_context.user_id}"):
                st.markdown(message["content"])
                st.caption(message["timestamp"])
        else:
            with st.chat_message("assistant", avatar="https://api.dicebear.com/7.x/bottts/svg?seed=travel-agent"):
                st.markdown(message["content"])
                st.caption(message["timestamp"])

    # User input
    user_input = st.chat_input("Let's plan a sustainable farming practice...")
    if user_input:
        handle_user_message(user_input)
        st.rerun()

    # Process message if needed
    if st.session_state.processing_message:
        user_input = st.session_state.processing_message
        st.session_state.processing_message = None
        
        # Process the message asynchronously
        with st.spinner("Thinking..."):
            try:
                # Prepare input for the agent using chat history
                if len(st.session_state.chat_history) > 1:
                    # Convert chat history to input list format for the agent
                    input_list = []
                    for msg in st.session_state.chat_history:
                        input_list.append({"role": msg["role"], "content": msg["content"]})
                else:
                    # First message
                    input_list = user_input

                # Display assistant response in chat message container
                response_content = ""
                
                # Create a chat message container using Streamlit's built-in component
                with st.chat_message("assistant", avatar="https://api.dicebear.com/7.x/bottts/svg?seed=AI-agent"):
                    message_placeholder = st.empty()
                    
                    # Run the async generator to fetch responses
                    async for chunk in invoke_agent_graph(user_input):
                        response_content += chunk
                        # Update only the text content
                        message_placeholder.markdown(response_content)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_content,
                    "timestamp": datetime.now().strftime("%I:%M %p")
                })
                
            except Exception as e:
                raise Exception(e)
                error_message = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_message,
                    "timestamp": datetime.now().strftime("%I:%M %p")
                })
            
            # Force a rerun to display the AI response
            st.rerun()

    # Footer
    st.divider()
    # st.caption("Powered by Pydantic AI and LangGraph | Built with Streamlit")

if __name__ == "__main__":
    asyncio.run(main())
