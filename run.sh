#!/bin/bash
# run.sh - Script to run the Code Librarian Agent using uv

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
CODEBASE_PATH="$(pwd)"  # Use current working directory by default
OUTPUT_DIR="${SCRIPT_DIR}/checkpoints"
VERBOSE=""

# Display help message
show_help() {
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  -c, --codebase-path PATH   Path to the codebase to analyze (default: current working directory)"
    echo "  -o, --output-dir PATH      Directory to save the checkpoint (default: ./checkpoints)"
    echo "  -v, --verbose              Enable verbose output"
    echo "  -h, --help                 Display this help message and exit"
    echo ""
    echo "Examples:"
    echo "  ./run.sh                   # Analyze the current directory"
    echo "  ./run.sh --codebase-path /path/to/your/codebase --output-dir ./my-checkpoints"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--codebase-path)
            CODEBASE_PATH="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if API keys are set
if [ -z "$GEMINI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: Either GEMINI_API_KEY or ANTHROPIC_API_KEY environment variable must be set."
    echo "Please set one of them using: export GEMINI_API_KEY=your_api_key_here"
    echo "or: export ANTHROPIC_API_KEY=your_api_key_here"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Ensure dependencies are installed
echo "Ensuring dependencies are installed..."
cd "$SCRIPT_DIR"

# Install dependencies with pip
echo "Installing dependencies with pip..."
pip install -r requirements.txt

# Run the Code Librarian Agent
echo "Running Code Librarian Agent on codebase: $CODEBASE_PATH"
echo "Output will be saved to: $OUTPUT_DIR"
echo ""

# Use python directly instead of uvx
python "${SCRIPT_DIR}/src/main.py" --codebase-path "$CODEBASE_PATH" --output-dir "$OUTPUT_DIR" $VERBOSE

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Code Librarian Agent completed successfully!"
    echo "Check $OUTPUT_DIR for the generated checkpoint."
else
    echo ""
    echo "Code Librarian Agent encountered an error."
    exit 1
fi
