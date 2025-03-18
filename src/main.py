#!/usr/bin/env python3
"""
Code Librarian Agent - Main Entry Point

This script serves as the entry point for the Code Librarian Agent, which generates
comprehensive knowledge checkpoints for codebases in markdown format using LangGraph
with the LATS (Language Agent Tree Search) pattern.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Set, TypedDict, Any, Union, Literal, Tuple, cast

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# Import LangGraph
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# Import tools
from tools.file_tools import ListDirectoryTool, ReadFileTool, SearchCodebaseTool
from tools.code_tools import AnalyzeCodeTool, GenerateArchitectureDiagramTool
from tools.security_tools import SecurityAuditTool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Node class for tree search
class Node:
    """Node in the LATS search tree."""
    
    def __init__(
        self,
        messages: List[BaseMessage],
        reflection: Optional[str] = None,
        parent: Optional['Node'] = None,
    ):
        """Initialize a node in the tree."""
        self.messages = messages
        self.reflection = reflection
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.is_solved = False
    
    def add_child(self, child: 'Node') -> None:
        """Add a child node to this node."""
        self.children.append(child)
        child.parent = self
    
    def update(self, value: float) -> None:
        """Update the value and visits count of this node."""
        self.value += value
        self.visits += 1
    
    @property
    def height(self) -> int:
        """Get the height of this node in the tree."""
        if self.parent is None:
            return 0
        return self.parent.height + 1
    
    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf node."""
        return len(self.children) == 0
    
    def get_best_child(self, c: float = 1.0) -> Optional['Node']:
        """Get the child with the highest UCT value."""
        if not self.children:
            return None
        
        def uct(node: Node) -> float:
            if node.visits == 0:
                return float("inf")
            exploitation = node.value / node.visits
            # Fixed UCT calculation using math.log and math.sqrt
            import math
            exploration = c * math.sqrt(2 * math.log(self.visits) / node.visits)
            return exploitation + exploration
        
        return max(self.children, key=uct)
    
    def get_best_trajectory(self) -> List[BaseMessage]:
        """Get the best trajectory from this node to a leaf node."""
        trajectory = []
        node = self
        while node:
            # Add the final AI message from each node
            ai_messages = [msg for msg in node.messages if isinstance(msg, AIMessage)]
            if ai_messages:
                trajectory.append(ai_messages[-1])
            # Move to the best child
            node = node.get_best_child()
        return trajectory

# Define the state schema using TypedDict
class TreeState(TypedDict, total=False):
    """State schema for the LATS agent."""
    root: Node
    input: str
    current_node: Node
    tools: List[Any]
    messages: List[Dict[str, Any]]
    iterations: int
    max_iterations: int
    task_complete: bool

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Code Librarian Agent")
    parser.add_argument("--codebase-path", type=str, default=os.getcwd(),
                      help="Path to the codebase to analyze")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Directory to save the checkpoint file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--max-iterations", type=int, default=10,
                      help="Maximum number of iterations for the search")
    parser.add_argument("--prompt-path", type=str, default=None,
                      help="Path to the prompt file")
    parser.add_argument("--model", type=str, default="gemma", choices=["gemma", "gemini", "claude"],
                      help="Model to use for analysis: gemma (Ollama local), gemini, or claude")
    parser.add_argument("--ollama-base-url", type=str, default="http://gruntus:11434/v1",
                      help="Base URL for Ollama API (only used with --model=gemma)")
    parser.add_argument("--ollama-model", type=str, default="gemma3:27b",
                      help="Model name for Ollama (only used with --model=gemma)")
    return parser.parse_args()

def initialize_tools(codebase_path):
    """Initialize the tools for the agent."""
    # File tools
    list_directory_tool = ListDirectoryTool(codebase_path=codebase_path)
    read_file_tool = ReadFileTool(codebase_path=codebase_path)
    search_codebase_tool = SearchCodebaseTool(codebase_path=codebase_path)
    
    # Code analysis tools
    analyze_code_tool = AnalyzeCodeTool(codebase_path=codebase_path)
    generate_architecture_diagram_tool = GenerateArchitectureDiagramTool(codebase_path=codebase_path)
    
    # Security tools
    security_audit_tool = SecurityAuditTool(codebase_path=codebase_path)
    
    return [
        list_directory_tool,
        read_file_tool,
        search_codebase_tool,
        analyze_code_tool,
        generate_architecture_diagram_tool,
        security_audit_tool
    ]

def initialize_llm(model="gemma", ollama_base_url="http://gruntus:11434/v1", ollama_model="gemma3:27b"):
    """Initialize the LLM based on the specified model and available API keys."""
    
    if model == "gemma":
        logger.info(f"Using Ollama with model {ollama_model}")
        try:
            # Use OpenAI compatible interface with Ollama
            return ChatOpenAI(
                model=ollama_model,
                temperature=0.2,
                base_url=ollama_base_url,
                api_key="ollama" # Placeholder, not actually used by Ollama
            )
        except Exception as e:
            logger.error(f"Error initializing Ollama model: {str(e)}")
            logger.error("Falling back to other available models")
            # Fall through to try other models
    
    if model == "claude" or os.environ.get("ANTHROPIC_API_KEY"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.error("Claude model requested but ANTHROPIC_API_KEY not found")
            if model == "claude":
                raise ValueError("ANTHROPIC_API_KEY environment variable is required when using --model=claude")
        else:
            logger.info("Using Anthropic Claude for analysis")
            return ChatAnthropic(
                model="claude-3-opus-20240229",
                temperature=0.2,
                anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
    
    if model == "gemini" or os.environ.get("GEMINI_API_KEY"):
        if not os.environ.get("GEMINI_API_KEY"):
            logger.error("Gemini model requested but GEMINI_API_KEY not found")
            if model == "gemini":
                raise ValueError("GEMINI_API_KEY environment variable is required when using --model=gemini")
        else:
            logger.info("Using Gemini for analysis")
            try:
                return ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro-latest",
                    temperature=0.2,
                    google_api_key=os.environ.get("GEMINI_API_KEY")
                )
            except Exception as e:
                logger.error(f"Error initializing Gemini model: {str(e)}")
                if "429" in str(e):
                    logger.error("CRITICAL ERROR: Gemini API quota exhausted (HTTP 429). Please try again later or use a different model.")
                    raise RuntimeError("Gemini API quota exhausted (HTTP 429). Please try again later or use a different model.")
                # Fall through to try other models if not explicitly requested
                if model == "gemini":
                    raise
    
    # If we get here and haven't returned a model yet, we have a problem
    raise ValueError("No suitable language model available. Please set up API keys or use a valid --model option.")

def load_prompt_template(prompt_path):
    """Load the system prompt template from a file."""
    try:
        with open(prompt_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found at {prompt_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading prompt file: {str(e)}")
        return None

def create_agent_prompt(codebase_path, prompt_path=None):
    """Create the system prompt for the agent."""
    # If a custom prompt path is provided, use it
    if prompt_path and os.path.exists(prompt_path):
        prompt_file_path = prompt_path
    else:
        # Try to find prompt.txt in the same directory as the run.sh script
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompt_file_path = os.path.join(script_dir, "prompt.txt")
    
    # Load the prompt template
    prompt_template = load_prompt_template(prompt_file_path)
    
    # If the prompt file doesn't exist or can't be read, use a minimal fallback prompt
    if not prompt_template:
        logger.warning("Using fallback minimal prompt since prompt.txt was not found")
        prompt_template = """You are an agent tasked with analyzing a codebase.
Use the available tools to explore and understand the codebase at: {codebase_path}"""
    
    # Format the prompt template with the codebase path
    # Use a dictionary-based format to handle various placeholders that might be in the prompt
    return prompt_template.format(codebase_path=codebase_path)

def generate_initial_response(state: TreeState) -> TreeState:
    """Generate the initial response for the agent."""
    logger.info("Generating initial response...")
    
    model = state.get("model", "gemma")
    ollama_base_url = state.get("ollama_base_url", "http://gruntus:11434/v1")
    ollama_model = state.get("ollama_model", "gemma3:27b")
    codebase_path = state["input"]
    
    llm = initialize_llm(model, ollama_base_url, ollama_model)
    tools = state["tools"]
    system_prompt = create_agent_prompt(codebase_path, state.get("prompt_path"))
    
    logger.info(f"Using prompt template for codebase analysis")
    
    # Create a ToolNode for handling tool calls
    tool_node = ToolNode(tools)
    
    # Initialize messages with system prompt
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Analyze the codebase at {codebase_path} and create a comprehensive knowledge checkpoint.")
    ]
    
    # Log messages being sent to LLM
    logger.info("=== SENDING MESSAGES TO LLM (INITIAL) ===")
    for i, msg in enumerate(messages):
        logger.info(f"Message {i+1} ({msg.type}): {msg.content[:200]}...")
    
    # Get initial response from LLM
    ai_message = llm.invoke(messages)
    
    # Log LLM response
    logger.info("=== LLM RESPONSE (INITIAL) ===")
    if isinstance(ai_message.content, str):
        logger.info(f"Response: {ai_message.content[:500]}...")
    else:
        logger.info(f"Response (non-string content): {str(ai_message.content)[:500]}...")
    
    messages.append(ai_message)
    
    # Log tool calls if present
    if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
        logger.info("=== TOOL CALLS IN RESPONSE ===")
        for i, tool_call in enumerate(ai_message.tool_calls):
            logger.info(f"Tool Call {i+1}: {tool_call}")
    
    # Process any tool calls in the response
    logger.info("Processing tool calls...")
    result = tool_node.invoke({"messages": messages})
    
    # Log detailed tool execution results
    if len(result.get("messages", [])) > len(messages):
        logger.info("=== TOOL EXECUTION RESULTS ===")
        for i, msg in enumerate(result.get("messages", [])[len(messages):]):
            logger.info(f"Tool Result {i+1} ({msg.type}): {str(msg.content)[:500]}...")
    
    messages = result.get("messages", messages)
    
    # Log tool results summary
    logger.info(f"Tool processing complete. Messages count: {len(messages)}")
    
    # Create root node with initial messages
    root_node = Node(messages=messages)
    
    # Update state
    return {
        **state,
        "root": root_node,
        "current_node": root_node,
        "messages": messages,
        "iterations": 1,
        "task_complete": False
    }

def should_continue(state: TreeState) -> Union[Literal["expand"], Literal["END"]]:
    """Determine whether to continue the tree search."""
    if state["task_complete"]:
        return END
    
    if state["iterations"] >= state["max_iterations"]:
        logger.info(f"Reached maximum iterations ({state['max_iterations']}). Ending search.")
        return END
    
    return "expand"

def expand_node(state: TreeState) -> TreeState:
    """Expand the current node by generating new actions."""
    logger.info(f"Expanding node (iteration {state['iterations']})...")
    
    model = state.get("model", "gemma")
    ollama_base_url = state.get("ollama_base_url", "http://gruntus:11434/v1")
    ollama_model = state.get("ollama_model", "gemma3:27b")
    
    llm = initialize_llm(model, ollama_base_url, ollama_model)
    tools = state["tools"]
    current_node = state["current_node"]
    messages = current_node.messages
    
    # Create a ToolNode for handling tool calls
    tool_node = ToolNode(tools)
    
    # Add a simple continuation message - let the LLM determine what to focus on next
    # based on its previous findings and the overall goal
    messages.append(HumanMessage(content="Continue your analysis and checkpoint creation. What else would you like to explore or document?"))
    
    # Log messages being sent to LLM
    logger.info(f"=== SENDING MESSAGES TO LLM (ITERATION {state['iterations']}) ===")
    # Log the last 3 messages to keep logs manageable
    for i, msg in enumerate(messages[-3:]):
        logger.info(f"Recent Message {len(messages)-3+i+1}/{len(messages)} ({msg.type}): {str(msg.content)[:200]}...")
    
    # Get next response from LLM
    ai_message = llm.invoke(messages)
    
    # Log LLM response
    logger.info(f"=== LLM RESPONSE (ITERATION {state['iterations']}) ===")
    if isinstance(ai_message.content, str):
        logger.info(f"Response: {ai_message.content[:500]}...")
    else:
        logger.info(f"Response (non-string content): {str(ai_message.content)[:500]}...")
    
    messages.append(ai_message)
    
    # Log tool calls if present
    if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
        logger.info("=== TOOL CALLS IN RESPONSE ===")
        for i, tool_call in enumerate(ai_message.tool_calls):
            logger.info(f"Tool Call {i+1}: {tool_call}")
    
    # Process any tool calls in the response
    logger.info("Processing tool calls...")
    result = tool_node.invoke({"messages": messages})
    
    # Log detailed tool execution results
    if len(result.get("messages", [])) > len(messages):
        logger.info("=== TOOL EXECUTION RESULTS ===")
        for i, msg in enumerate(result.get("messages", [])[len(messages):]):
            logger.info(f"Tool Result {i+1} ({msg.type}): {str(msg.content)[:500]}...")
    
    messages = result.get("messages", messages)
    
    # Log tool results summary
    logger.info(f"Tool processing complete. Messages count: {len(messages)}")
    
    # Create a new child node
    child_node = Node(messages=messages)
    current_node.add_child(child_node)
    
    # Check if the agent has completed the task by producing a properly structured checkpoint
    # based on the template provided in the prompt
    checkpoint_complete = False
    for message in reversed(messages[-5:]):  # Check last few messages
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                # Look for the required sections from the checkpoint template
                required_sections = [
                    "## Codebase Overview",
                    "## User Experience",
                    "## Software Architecture",
                    "```mermaid",
                    "## Deployment Architecture",
                    "## CI/CD Pipeline",
                    "## Class Diagrams",
                    "## Unexpected Observations",
                    "## Security Audit"
                ]
                
                # Count how many required sections are present
                sections_present = sum(1 for section in required_sections if section in content)
                
                # Check if most of the required sections are present (at least 7 of 9)
                if sections_present >= 7:
                    checkpoint_complete = True
                    logger.info(f"Agent has produced a structured checkpoint with {sections_present}/9 required sections.")
                    break
                elif ("COMPLETE" in content.upper() or 
                      "FINISHED" in content.upper() or 
                      "COMPLETED ANALYSIS" in content.upper() or
                      "FINAL CHECKPOINT" in content.upper()):
                    logger.info("Agent indicates task completion, but structured output may be incomplete.")
                    # Only mark as complete if at least some structured sections are present
                    if sections_present >= 3:
                        checkpoint_complete = True
                        break
    
    # Update state
    return {
        **state,
        "current_node": child_node,
        "messages": messages,
        "iterations": state["iterations"] + 1,
        "task_complete": checkpoint_complete
    }

def create_lats_graph(tools: List[Any], max_iterations: int, model: str = "gemma", 
                   ollama_base_url: str = "http://gruntus:11434/v1", 
                   ollama_model: str = "gemma3:27b") -> StateGraph:
    """Create the LATS graph."""
    
    # Define initial state
    initial_state: TreeState = {
        "root": None,  # Will be populated in the first step
        "input": "",   # Will be populated when running the graph
        "current_node": None,  # Will be populated in the first step
        "tools": tools,
        "messages": [],
        "iterations": 0,
        "max_iterations": max_iterations,
        "task_complete": False
    }
    
    # Create the graph
    builder = StateGraph(TreeState)
    
    # Add nodes
    builder.add_node("start", generate_initial_response)
    builder.add_node("expand", expand_node)
    
    # Set entry point
    builder.set_entry_point("start")
    
    # Add conditional edges
    builder.add_conditional_edges(
        "start",
        should_continue,
    )
    builder.add_conditional_edges(
        "expand",
        should_continue,
    )
    
    # Compile the graph
    return builder.compile()

def extract_final_response(state: TreeState) -> str:
    """Extract the final response from the tree state."""
    logger.info("Extracting final response from the search tree...")
    
    # Required sections from the checkpoint template
    required_sections = [
        "## Codebase Overview",
        "## User Experience",
        "## Software Architecture",
        "```mermaid",
        "## Deployment Architecture",
        "## CI/CD Pipeline",
        "## Class Diagrams",
        "## Unexpected Observations",
        "## Security Audit"
    ]
    
    # Function to score a response based on content quality
    def score_response(content):
        if not isinstance(content, str):
            return -1
            
        # Count required sections
        sections_present = sum(1 for section in required_sections if section in content)
        
        # Base score is the number of sections present
        score = sections_present * 10
        
        # Add points for response length (but cap it to avoid favoring extreme verbosity)
        length_score = min(len(content) / 500, 20)
        score += length_score
        
        # Add points for mermaid diagrams (they're important!)
        mermaid_sections = content.count("```mermaid")
        score += mermaid_sections * 5
        
        return score
    
    # Get all responses from all nodes
    all_responses = []
    
    # First, collect responses from the current node
    current_node = state["current_node"]
    for message in current_node.messages:
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str) and len(content) > 100:  # Minimum length threshold
                all_responses.append((content, score_response(content)))
    
    # If we need more options, check all nodes
    if not all_responses or max(score for _, score in all_responses) < 50:
        logger.info("Searching all nodes for structured responses...")
        # BFS through all nodes
        queue = [state["root"]]
        while queue:
            node = queue.pop(0)
            queue.extend(node.children)
            
            # Skip the current node as we've already processed it
            if node == current_node:
                continue
                
            for message in node.messages:
                if isinstance(message, AIMessage):
                    content = message.content
                    if isinstance(content, str) and len(content) > 100:  # Minimum length threshold
                        all_responses.append((content, score_response(content)))
    
    # Sort responses by score (highest first)
    all_responses.sort(key=lambda x: x[1], reverse=True)
    
    # Log scores for top responses for debugging
    for i, (content, score) in enumerate(all_responses[:3]):
        if i < len(all_responses):
            sections = sum(1 for section in required_sections if section in content)
            mermaid_count = content.count("```mermaid")
            logger.info(f"Response #{i+1}: Score={score:.1f}, Sections={sections}/9, Mermaid={mermaid_count}, Length={len(content)}")
    
    # Get the best response
    if all_responses:
        best_response, best_score = all_responses[0]
        logger.info(f"Selected best response with score: {best_score:.1f}")
    else:
        logger.warning("No suitable AI message found. Using fallback.")
        best_response = "# Analysis of Codebase\n\nThe agent completed the analysis but did not produce a structured checkpoint according to the template."
    
    return best_response

def main():
    """Main function for the Code Librarian Agent."""
    # Parse arguments
    args = parse_arguments()
    codebase_path = args.codebase_path
    output_dir = args.output_dir
    verbose = args.verbose
    max_iterations = args.max_iterations
    prompt_path = args.prompt_path
    model = args.model
    ollama_base_url = args.ollama_base_url
    ollama_model = args.ollama_model
    
    # Check if the codebase path exists
    if not os.path.exists(codebase_path):
        logger.error(f"Codebase path {codebase_path} does not exist.")
        sys.exit(1)
    logger.info(f"Analyzing codebase at {codebase_path}")
    
    # Initialize tools
    tools = initialize_tools(codebase_path)
    
    # Create the LATS graph
    graph = create_lats_graph(tools, max_iterations, model, ollama_base_url, ollama_model)
    
    # Run the agent
    try:
        logger.info(f"Starting code analysis using LATS with model {model} (this may take some time)...")
        
        # Run the graph with the codebase path as input
        result = graph.invoke({
            "input": codebase_path,
            "tools": tools,
            "max_iterations": max_iterations,
            "prompt_path": prompt_path,
            "model": model,
            "ollama_base_url": ollama_base_url,
            "ollama_model": ollama_model
        })
        
        # Extract the final response
        final_response = extract_final_response(result)
        
        # Save the checkpoint with more unique timestamp (date and time)
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        output_file = f"{timestamp}-checkpoint.md"
        
        # Prioritize saving in the codebase directory
        codebase_output_path = os.path.join(codebase_path, output_file)
        try:
            with open(codebase_output_path, "w") as f:
                f.write(final_response)
            logger.info(f"Checkpoint saved to {codebase_output_path}")
        except Exception as e:
            logger.warning(f"Could not save to codebase directory: {str(e)}")
            
            # Fall back to output directory if specified
            if output_dir:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_path = os.path.join(output_dir, output_file)
                with open(output_path, "w") as f:
                    f.write(final_response)
                logger.info(f"Checkpoint saved to fallback location: {output_path}")
            else:
                # Generate a default output file in current directory
                with open(output_file, "w") as f:
                    f.write(final_response)
                logger.info(f"Checkpoint saved to fallback location: {output_file}")
        
        # Print the final response
        print(final_response)
        
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        sys.exit(1)

# Check if the script is being run directly
if __name__ == "__main__":
    # Run the main function
    main()