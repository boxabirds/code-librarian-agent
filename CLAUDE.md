# Code Librarian Agent - Development Guidelines

## Build and Run Commands
- Run the agent: `python src/main.py --codebase-path /path/to/codebase --output-dir ./checkpoints`
- Run with shell script: `./run.sh --codebase-path /path/to/codebase --output-dir ./checkpoints --verbose`
- Install dependencies: `pip install -r requirements.txt`
- Required environment variable: `GEMINI_API_KEY` (preferred) or `ANTHROPIC_API_KEY`

## Project Purpose
The Code Librarian Agent analyzes codebases and generates comprehensive knowledge checkpoints in markdown format that include:
- Purpose statement
- User experience narrative
- Architecture diagrams (software, deployment, CI/CD)
- Class diagrams showing entity relationships
- Unexpected observations
- Security audit

## Code Style Guidelines
- **Formatting**: Follow PEP 8 conventions
- **Imports**: Standard library first, then third-party, then local modules
- **Types**: Use type annotations with `typing` module (TypedDict, List, Dict, etc.)
- **Naming**: 
  - snake_case for functions, methods, variables
  - CamelCase for classes
  - UPPER_CASE for constants
- **Error Handling**: Use try/except blocks with specific exceptions
- **Documentation**: Use docstrings for modules, classes, and functions
- **Paths**: Always use relative paths within the codebase, never absolute paths

## Project Structure
- `/src`: Main source code
- `/src/tools`: Custom tools for code analysis
- `/src/templates`: Output templates
- `/checkpoints`: Generated analysis files