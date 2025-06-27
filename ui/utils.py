from typing import Any, Callable, Dict, List, Optional, Union

import streamlit as st
from agno.agent import Agent
from agno.document import Document
from agno.document.reader import Reader
from agno.document.reader.csv_reader import CSVReader
from agno.document.reader.docx_reader import DocxReader
from agno.document.reader.pdf_reader import PDFReader
from agno.document.reader.text_reader import TextReader
from agno.document.reader.website_reader import WebsiteReader
from agno.utils.log import logger


async def initialize_agent_session_state(agent_name: str):
    logger.info(f"---*--- Initializing session state for {agent_name} ---*---")
    st.session_state[agent_name] = {
        "agent": None,
        "session_id": None,
        "messages": [],
    }


async def initialize_team_session_state(team_name: str):
    logger.info(f"---*--- Initializing session state for {team_name} ---*---")
    st.session_state[team_name] = {
        "team": None,
        "session_id": None,
        "messages": [],
    }


async def initialize_workflow_session_state(workflow_name: str):
    logger.info(f"---*--- Initializing session state for {workflow_name} ---*---")
    st.session_state[workflow_name] = {
        "workflow": None,
        "session_id": None,
        "messages": [],
    }


async def selected_model() -> str:
    """Display a model selector in the sidebar."""
    model_options = {
        "gpt-4o": "gpt-4o",
        "o3-mini": "o3-mini",
    }
    selected_model = st.sidebar.selectbox(
        "Choose a model",
        options=list(model_options.keys()),
        index=0,
        key="model_selector",
    )
    return model_options[selected_model]


async def add_message(
    agent_name: str,
    role: str,
    content: str,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Safely add a message to the Agent's session state."""
    # if role == "user":
    #     logger.info(f"👤  {role} → {agent_name}: {content}")
    # else:
    #     logger.info(f"🤖  {agent_name} → user: {content}")
    st.session_state[agent_name]["messages"].append({"role": role, "content": content, "tool_calls": tool_calls})


def display_tool_calls(tool_calls_container, tools):
    """Display tool calls in a streamlit container with expandable sections.

    Args:
        tool_calls_container: Streamlit container to display the tool calls
        tools: List of tool call dictionaries containing name, args, content, and metrics
    """
    if not tools:
        return

    try:
        with tool_calls_container.container():
            for tool_call in tools:
                if hasattr(tool_call, 'tool_name'):
                    # Handle object with attributes
                    tool_name = tool_call.tool_name
                    tool_args = tool_call.tool_args
                    content = tool_call.result if hasattr(tool_call, 'result') else None
                    metrics = getattr(tool_call, "metrics", None)
                else:
                    # Handle dictionary
                    tool_name = tool_call.get("tool_name", "Unknown Tool")
                    tool_args = tool_call.get("tool_args", {})
                    content = tool_call.get("content")
                    metrics = tool_call.get("metrics", {})

                # Add timing information
                execution_time_str = "N/A"
                try:
                    if metrics:
                        execution_time = metrics.time
                        if execution_time is not None:
                            execution_time_str = f"{execution_time:.2f}s"
                except Exception as e:
                    logger.error(f"Error displaying tool calls: {str(e)}")
                    pass

                with st.expander(
                    f"🛠️ {tool_name.replace('_', ' ').title() if tool_name else 'Tool'} ({execution_time_str})",
                    expanded=False,
                ):
                    # Show query with syntax highlighting
                    if isinstance(tool_args, dict) and tool_args.get("query"):
                        st.code(tool_args["query"], language="sql")

                    # Display arguments in a more readable format
                    if tool_args and tool_args != {"query": None}:
                        st.markdown("**Arguments:**")
                        st.json(tool_args)

                    if content:
                        st.markdown("**Results:**")
                        try:
                            # Check if content is already a dictionary or can be parsed as JSON
                            if isinstance(content, dict) or (
                                isinstance(content, str) and content.strip().startswith(("{", "["))
                            ):
                                st.json(content)
                            else:
                                # If not JSON, show as markdown
                                st.markdown(content)
                        except Exception:
                            # If JSON display fails, show as markdown
                            st.markdown(content)
    except Exception as e:
        logger.error(f"Error displaying tool calls: {str(e)}")
        tool_calls_container.error(f"Failed to display tool results: {str(e)}")


async def example_inputs(agent_name: str) -> None:
    """Show example inputs for an Agent."""
    with st.sidebar:
        st.markdown("#### :thinking_face: Try me!")
        if st.button("Who are you?"):
            await add_message(
                agent_name,
                "user",
                "Who are you?",
            )
        if st.button("What is your purpose?"):
            await add_message(
                agent_name,
                "user",
                "What is your purpose?",
            )

        # Agent-specific examples
        if agent_name == "sage":
            if st.button("Tell me about Agno"):
                await add_message(
                    agent_name,
                    "user",
                    "Tell me about Agno. Github repo: https://github.com/agno-agi/agno. Documentation: https://docs.agno.com",
                )
        elif agent_name == "scholar":
            if st.button("Tell me about the US tariffs"):
                await add_message(
                    agent_name,
                    "user",
                    "Tell me about the US tariffs",
                )


async def knowledge_widget(agent_name: str, agent: Agent) -> None:
    """Display a knowledge widget in the sidebar."""

    if agent is not None and agent.knowledge is not None:
        # Add websites to knowledge base
        if "url_scrape_key" not in st.session_state:
            st.session_state[agent_name]["url_scrape_key"] = 0
        input_url = st.sidebar.text_input(
            "Add URL to Knowledge Base", type="default", key=st.session_state[agent_name]["url_scrape_key"]
        )
        add_url_button = st.sidebar.button("Add URL")
        if add_url_button:
            if input_url is not None:
                alert = st.sidebar.info("Processing URLs...", icon="ℹ️")
                if f"{input_url}_scraped" not in st.session_state:
                    scraper = WebsiteReader(max_links=2, max_depth=1)
                    web_documents: List[Document] = scraper.read(input_url)
                    if web_documents:
                        agent.knowledge.load_documents(web_documents, upsert=True)
                    else:
                        st.sidebar.error("Could not read website")
                    st.session_state[f"{input_url}_uploaded"] = True
                alert.empty()

        # Add documents to knowledge base
        if "file_uploader_key" not in st.session_state:
            st.session_state[agent_name]["file_uploader_key"] = 100
        uploaded_file = st.sidebar.file_uploader(
            "Add Document to Knowledge Base",
            type=["pdf", "docx", "txt", "csv"],
            key=st.session_state[agent_name]["file_uploader_key"],
        )
        if uploaded_file is not None:
            alert = st.sidebar.info("Processing document...", icon="ℹ️")
            if f"{uploaded_file.name}_uploaded" not in st.session_state:
                try:
                    # Determine the appropriate reader based on file type
                    if uploaded_file.name.endswith(".pdf"):
                        reader: Reader = PDFReader()
                    elif uploaded_file.name.endswith(".docx"):
                        reader = DocxReader()
                    elif uploaded_file.name.endswith(".csv"):
                        reader = CSVReader()
                    else:
                        reader = TextReader()

                    # Read the document
                    documents: List[Document] = reader.read(uploaded_file)
                    if documents:
                        agent.knowledge.load_documents(documents, upsert=True)
                        st.sidebar.success(f"✅ {uploaded_file.name} added to knowledge base!")
                    else:
                        st.sidebar.error(f"❌ Could not read {uploaded_file.name}")
                    st.session_state[f"{uploaded_file.name}_uploaded"] = True
                except Exception as e:
                    st.sidebar.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            alert.empty()


async def session_selector(agent_name: str, agent: Agent, get_agent: Callable, user_id: str, model_id: str) -> None:
    """Display a session selector in the sidebar."""
    with st.sidebar:
        st.markdown("#### :gear: Session Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Session"):
                restart_agent(agent_name)
        with col2:
            if st.button("��️ Clear History"):
                if agent_name in st.session_state:
                    st.session_state[agent_name]["messages"] = []
                st.rerun()

        # Export chat history
        if agent_name in st.session_state and st.session_state[agent_name]["messages"]:
            chat_history = export_chat_history(agent_name)
            st.download_button(
                label="📥 Export Chat",
                data=chat_history,
                file_name=f"{agent_name}_chat_history.md",
                mime="text/markdown",
            )

        # Session ID display
        if agent_name in st.session_state and st.session_state[agent_name]["session_id"]:
            st.markdown(f"**Session ID:** `{st.session_state[agent_name]['session_id']}`")


def export_chat_history(agent_name: str):
    """Export chat history as markdown."""
    if agent_name not in st.session_state:
        return ""

    messages = st.session_state[agent_name]["messages"]
    chat_text = f"# {agent_name.title()} Chat History\n\n"
    chat_text += f"**Date:** {st.session_state[agent_name].get('session_id', 'Unknown')}\n\n"

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "user":
            chat_text += f"## �� User\n\n{content}\n\n"
        elif role == "assistant":
            chat_text += f"## 🤖 {agent_name.title()}\n\n{content}\n\n"
            
            # Add tool calls if present
            if msg.get("tool_calls"):
                chat_text += "#### Tool Calls:\n"
                for i, tool_call in enumerate(msg["tool_calls"]):
                    if hasattr(tool_call, 'tool_name'):
                        tool_name = tool_call.tool_name
                        chat_text += f"**{i + 1}. {tool_name}**\n\n"
                    else:
                        tool_name = tool_call.get("tool_name", "Unknown Tool")
                        chat_text += f"**{i + 1}. {tool_name}**\n\n"
                        
                        # Add tool arguments if available
                        tool_args = tool_call.get("tool_args", {})
                        if tool_args:
                            chat_text += f"**Arguments:**\n```json\n{tool_args}\n```\n\n"
                        
                        # Add tool results if available
                        tool_content = tool_call.get("content")
                        if tool_content:
                            chat_text += f"**Results:**\n```\n{tool_content}\n```\n\n"

    return chat_text


async def utilities_widget(agent_name: str, agent: Agent) -> None:
    """Display utilities widget in the sidebar."""
    with st.sidebar:
        st.markdown("#### :wrench: Utilities")
        if st.button("🔄 Restart Agent"):
            restart_agent(agent_name)
            st.rerun()


def restart_agent(agent_name: str):
    """Restart the agent by clearing session state."""
    if agent_name in st.session_state:
        st.session_state[agent_name] = {
            "agent": None,
            "session_id": None,
            "messages": [],
        }


async def about_agno():
    """Show information about Agno in the sidebar"""
    with st.sidebar:
        st.markdown("### About Agno ✨")
        st.markdown("""
        Agno is an open-source library for building Multimodal Agents.

        [GitHub](https://github.com/agno-agi/agno) | [Docs](https://docs.agno.com)
        """)

        st.markdown("### Need Help?")
        st.markdown(
            "If you have any questions, catch us on [discord](https://agno.link/discord) or post in the community [forum](https://agno.link/community)."
        )


async def footer():
    st.markdown("---")
    st.markdown(
        "<p style='text-align: right; color: gray;'>Built using <a href='https://github.com/agno-agi/agno'>Agno</a></p>",
        unsafe_allow_html=True,
    )
