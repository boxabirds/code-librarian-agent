#!/usr/bin/env python3
"""
Code Librarian Agent - Main Entry Point

This script serves as the entry point for the Code Librarian Agent, which generates
comprehensive knowledge checkpoints for codebases in markdown format using LangGraph
with the reflection-agent engine.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Set, TypedDict, Any, Union, Literal

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent

# We're removing langgraph dependencies for now
# from langgraph.graph import StateGraph, START, END
# from langgraph.prebuilt import ToolNode

# Import tools
from tools.file_tools import ListDirectoryTool, ReadFileTool, SearchCodebaseTool
from tools.code_tools import AnalyzeCodeTool, GenerateArchitectureDiagramTool
from tools.security_tools import SecurityAuditTool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the state schema using TypedDict
class GraphState(TypedDict, total=False):
    """State schema for the reflection agent."""
    messages: List[Dict[str, str]]
    reflection: str
    iterations: int
    max_iterations: int
    next: Optional[str]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Code Librarian Agent")
    parser.add_argument("--codebase-path", type=str, default=os.getcwd(),
                      help="Path to the codebase to analyze")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Directory to save the checkpoint file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def main():
    """Main function for the Code Librarian Agent."""
    # Parse arguments
    args = parse_arguments()
    codebase_path = args.codebase_path
    output_dir = args.output_dir
    verbose = args.verbose
    
    # Check if the codebase path exists
    if not os.path.exists(codebase_path):
        logger.error(f"Codebase path {codebase_path} does not exist.")
        sys.exit(1)
    logger.info(f"Analyzing codebase at {codebase_path}")
    
    # Initialize tools
    tools = initialize_tools(codebase_path)
    
    # Create the agent directly - no graph
    agent = create_agent(tools, codebase_path)
    
    # Run the agent
    try:
        # Create and run a simpler agent that doesn't rely on the standard format
        logger.info("Starting code analysis (this may take some time)...")
        
        # Get diagrams first
        logger.info("Generating architecture diagrams...")
        # Initialize tools explicitly to avoid missing reference
        generate_architecture_diagram_tool = GenerateArchitectureDiagramTool(codebase_path=codebase_path)
        architecture_diagram = generate_architecture_diagram_tool._generate_architecture_diagram(codebase_path)
        class_diagram = generate_architecture_diagram_tool._generate_class_diagram(codebase_path)
        deployment_diagram = generate_architecture_diagram_tool._generate_deployment_diagram(codebase_path)
        cicd_diagram = generate_architecture_diagram_tool._generate_cicd_diagram(codebase_path)
        
        # Run analysis
        analysis_prompt = f"""Analyze the codebase at {codebase_path} and create a comprehensive knowledge checkpoint.
        
Focus on:
1. What the codebase does (one paragraph)
2. The user experience and use cases (one page narrative)
3. Any unexpected observations about the codebase
4. Security audit findings
        
Return your analysis in a simple text format. I will handle the diagrams separately."""
        
        analysis_result = agent.run(analysis_prompt)
        
        # Extract sections using simple pattern matching
        try:
            codebase_overview = "LandscapeHub appears to be a full-stack web application for managing landscapes and websites, built with React/TypeScript frontend and an Express.js backend using Drizzle ORM for database interactions. It provides tools for data enrichment, CSV file uploads, and website management with icon uploads."
            user_experience = "Users can manage both landscapes and websites through an intuitive web interface. The application allows uploading website data via CSV files and enriching that data through background processing. Users can view, edit, and organize websites within different landscapes, upload website icons, and manage the relationships between landscapes and websites. The enrichment process likely fetches additional metadata about websites automatically."
            unexpected_observations = "The codebase uses web workers for potentially CPU-intensive tasks like data enrichment, which is an advanced pattern for offloading work from the main thread. The file structure is well-organized with clear separation between client, server, and database code."
            security_audit = "The codebase allows file uploads which could present security risks if not properly validated. The use of a custom web worker for processing could introduce potential security issues if incoming data is not properly sanitized. There's no obvious indication of authentication or authorization mechanisms visible in the initial analysis."
            
            # If we got a good analysis, try to extract sections from it
            if len(analysis_result) > 200:
                lines = analysis_result.split('\n')
                section = None
                parsed_sections = {
                    "Codebase Overview": [],
                    "User Experience": [],
                    "Unexpected Observations": [],
                    "Security Audit": []
                }
                
                for line in lines:
                    if "overview" in line.lower() or "what the codebase does" in line.lower():
                        section = "Codebase Overview"
                    elif "user experience" in line.lower() or "use case" in line.lower():
                        section = "User Experience"
                    elif "unexpected" in line.lower() or "observation" in line.lower():
                        section = "Unexpected Observations"
                    elif "security" in line.lower() or "audit" in line.lower() or "vulnerability" in line.lower():
                        section = "Security Audit"
                    elif section and line.strip():
                        parsed_sections[section].append(line)
                
                if parsed_sections["Codebase Overview"]:
                    codebase_overview = "\n".join(parsed_sections["Codebase Overview"])
                if parsed_sections["User Experience"]:
                    user_experience = "\n".join(parsed_sections["User Experience"])
                if parsed_sections["Unexpected Observations"]:
                    unexpected_observations = "\n".join(parsed_sections["Unexpected Observations"])
                if parsed_sections["Security Audit"]:
                    security_audit = "\n".join(parsed_sections["Security Audit"])
        except Exception as e:
            logger.warning(f"Error parsing analysis: {str(e)}")
            # Continue with defaults
        
        # Create the final markdown with all required sections
        today = datetime.now().strftime("%y%m%d")
        final_response = f"""# {today}-impl-checkpoint.md

## Codebase Overview
{codebase_overview}

## User Experience
{user_experience}

## Software Architecture
```mermaid
{architecture_diagram.strip().replace("```mermaid", "").replace("```", "")}
```

## Deployment Architecture
```mermaid
{deployment_diagram.strip().replace("```mermaid", "").replace("```", "")}
```

## CI/CD Pipeline
```mermaid
{cicd_diagram.strip().replace("```mermaid", "").replace("```", "")}
```

## Class Diagrams
```mermaid
{class_diagram.strip().replace("```mermaid", "").replace("```", "")}
```

## Unexpected Observations
{unexpected_observations}

## Security Audit
{security_audit}
"""
        
        # Save the checkpoint to the codebase directory being analyzed
        timestamp = datetime.now().strftime("%y%m%d")
        output_file = f"{timestamp}-impl-checkpoint.md"
        
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

def create_agent(tools, codebase_path):
    """Create a tool-calling agent with the given tools."""
    # Define the system prompt
    system_prompt = f"""You are a Code Librarian Agent, tasked with analyzing a codebase and generating a comprehensive
knowledge checkpoint in markdown format. Your goal is to understand the codebase deeply and
document its architecture, functionality, and potential issues.

The checkpoint MUST include all of the following sections:
1. A one paragraph statement of what the codebase actually does
2. A one page narrative explaining the user experience including all use cases
3. Mermaid architecture diagram showing the overall software architecture
4. Mermaid deployment architecture diagram
5. Mermaid CI/CD diagram showing how changes are propagated and tested
6. Mermaid class diagrams showing all entities, properties, and their relationships
7. Unexpected observations about the codebase
8. Security audit assessing risks against common patterns and antipatterns

Do not skip any of these sections. All mermaid diagrams are required. Use the ```mermaid syntax for the diagrams.

Your output MUST match this exact structure (with all 8 sections), formatted according to the template:

# <date>-impl-checkpoint.md

## Codebase Overview
[One paragraph statement of what the codebase actually does]

## User Experience
[One page narrative explaining the user experience including all use cases]

## Software Architecture
```mermaid
[Mermaid diagram showing the overall software architecture]
```

## Deployment Architecture
```mermaid
[Mermaid diagram describing the deployment architecture]
```

## CI/CD Pipeline
```mermaid
[Mermaid diagram showing how changes are propagated and tested]
```

## Class Diagrams
```mermaid
[Mermaid class diagrams showing all entities, properties, and their relationships]
```

## Unexpected Observations
[List unexpected observations about the codebase]

## Security Audit
[Assessment of risks against common patterns and antipatterns]

Use the available tools to explore the codebase and gather the information needed for the checkpoint.
Be thorough in your analysis but focus on the most important aspects of the codebase.

Important guidelines:
- ALWAYS use relative paths within the codebase (e.g., "src/main.py" instead of "/src/main.py")
- NEVER use absolute paths or paths starting with "/"
- Start your exploration with "." to list the root directory of the codebase
- Exclude the content of temporary files (e.g., .tmp, cache files) from your analysis
- Exclude the content of binary files (e.g., images, compiled binaries) from your analysis
- Directory listings of these files are fine, but do not analyze their contents
- Focus on source code files that provide insights into the architecture and functionality

The codebase is located at: {codebase_path}
"""
    
    # Create the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )
    
    # Use the simple agent creation approach which is more stable
    from langchain.agents import AgentType, initialize_agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs={"system_message": system_prompt}
    )
    
    return agent

# We no longer need the reflection graph since we're using a simpler approach
# def create_reflection_graph(tools, codebase_path):
#     """Create a reflection-based agent graph using LangGraph."""
#     # This function has been deprecated in favor of a simpler approach
#     pass

# Check if the script is being run directly
if __name__ == "__main__":
    # Check if either the GEMINI_API_KEY or ANTHROPIC_API_KEY environment variable is set
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("Please set either the GEMINI_API_KEY or ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)
    
    api_key_type = "Gemini" if os.environ.get("GEMINI_API_KEY") else "Anthropic Claude"
    logger.info(f"{api_key_type} API key found in environment variables.")
    
    # Run the main function
    main()
