# Code Librarian Agent

A reflection-based agent for taking comprehensive design snapshots of codebases. The agent analyzes a codebase and generates a markdown checkpoint document containing:

- A concise statement of what the codebase does
- A narrative explaining the user experience and use cases
- Mermaid architecture diagrams (software architecture, deployment, CI/CD)
- Class diagrams showing entities and their relationships
- Unexpected observations about the codebase
- Security audit assessing risks against common patterns and antipatterns

## Requirements

- Python 3.9+
- Google Gemini API key (preferred) or Anthropic API key (Claude)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/code-librarian-agent.git
   cd code-librarian-agent
   ```

2. Install dependencies using uvx:
   ```
   uvx pip install -r requirements.txt
   ```

3. Set your API key:
   ```
   # Recommended
   export GEMINI_API_KEY=your_gemini_api_key_here
   
   # Alternatively
   export ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Usage

Run the code librarian agent on a codebase:

```
python src/main.py --codebase-path /path/to/your/codebase --output-dir ./checkpoints
```

### Options

- `--codebase-path`: Path to the codebase to analyze (default: current directory)
- `--output-dir`: Directory to save the checkpoint (default: ./checkpoints)
- `--verbose`: Enable verbose output

## How It Works

The Code Librarian Agent uses LangGraph's reflection-agent engine with Google Gemini (or optionally Claude) to:

1. Explore and analyze the codebase structure
2. Identify key components, patterns, and relationships
3. Generate Mermaid diagrams to visualize the architecture
4. Perform a security audit to identify potential vulnerabilities
5. Compile all findings into a comprehensive markdown checkpoint

The agent uses a reflection mechanism to continuously evaluate its progress and decide what aspects of the codebase to investigate next.

## Architecture

The project is structured as follows:

```
code-librarian-agent/
├── src/
│   ├── main.py              # Entry point
│   ├── tools/               # Custom tools
│   │   ├── file_tools.py    # File operations
│   │   ├── code_tools.py    # Code analysis tools
│   │   └── security_tools.py # Security audit tools
│   └── templates/           # Templates for output
│       └── checkpoint.md.j2 # Jinja2 template for checkpoint
├── checkpoints/             # Generated checkpoints
└── requirements.txt         # Dependencies
```

## License

[MIT License](LICENSE)
