"""
File Tools

This module provides tools for interacting with the file system to analyze a codebase.
"""

import os
import glob
from typing import Dict, List, Optional, Any, ClassVar, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class ListDirectoryTool(BaseTool):
    """Tool for listing the contents of a directory in the codebase."""
    
    name: ClassVar[str] = "list_directory"
    description: ClassVar[str] = "Lists files and directories at the specified path within the codebase."
    codebase_path: str
    
    def __init__(self, codebase_path: str):
        """
        Initialize the ListDirectoryTool.
        
        Args:
            codebase_path: Absolute path to the codebase
        """
        super().__init__(codebase_path=codebase_path)
        self.codebase_path = codebase_path
    
    def _run(self, path: str = ".", include_hidden: bool = False) -> List[Dict[str, Any]]:
        """
        List the contents of a directory within the codebase.
        
        Args:
            path: Relative path within the codebase
            include_hidden: Whether to include hidden files and directories
            
        Returns:
            List of dictionaries with file/directory information
        """
        # Normalize the path to prevent path traversal
        norm_path = os.path.normpath(path)
        
        # Ensure the path doesn't start with / or contain ..
        if norm_path.startswith('/') or norm_path.startswith('\\') or '..' in norm_path.split(os.sep):
            return [{"error": f"Invalid path: {path}. Path must be relative to the codebase."}]
        
        target_path = os.path.join(self.codebase_path, norm_path)
        
        # Ensure the target path is within the codebase
        if not os.path.abspath(target_path).startswith(os.path.abspath(self.codebase_path)):
            return [{"error": f"Path traversal attempt detected: {path}. Path must be within the codebase."}]
        
        if not os.path.exists(target_path):
            return [{"error": f"Path does not exist: {path}"}]
        
        if not os.path.isdir(target_path):
            return [{"error": f"Path is not a directory: {path}"}]
        
        results = []
        
        for item in os.listdir(target_path):
            # Skip hidden files and directories if not included
            if not include_hidden and item.startswith("."):
                continue
            
            item_path = os.path.join(target_path, item)
            is_dir = os.path.isdir(item_path)
            
            # Get file size or count files in directory
            if is_dir:
                size = None
                item_type = "directory"
                try:
                    # Count files in directory (recursive)
                    file_count = sum([len(files) for _, _, files in os.walk(item_path)])
                    dir_count = sum([len(dirs) for _, dirs, _ in os.walk(item_path)])
                    child_count = file_count + dir_count
                except:
                    child_count = None
            else:
                try:
                    size = os.path.getsize(item_path)
                except:
                    size = None
                item_type = "file"
                child_count = None
            
            # Get file extension
            _, ext = os.path.splitext(item)
            ext = ext[1:] if ext else None
            
            results.append({
                "name": item,
                "path": os.path.join(path, item),
                "type": item_type,
                "size": size,
                "extension": ext,
                "child_count": child_count
            })
        
        # Sort results: directories first, then files, both alphabetically
        results.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))
        
        return results


class ReadFileTool(BaseTool):
    """Tool for reading the contents of a file in the codebase."""
    
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = "Reads the contents of a file within the codebase."
    codebase_path: str
    
    def __init__(self, codebase_path: str):
        """
        Initialize the ReadFileTool.
        
        Args:
            codebase_path: Absolute path to the codebase
        """
        super().__init__(codebase_path=codebase_path)
        self.codebase_path = codebase_path
    
    def _run(self, path: str, max_length: Optional[int] = None) -> Dict[str, Any]:
        """
        Read the contents of a file within the codebase.
        
        Args:
            path: Relative path to the file within the codebase
            max_length: Maximum number of characters to read
            
        Returns:
            Dictionary with file content or error message
        """
        # Normalize the path to prevent path traversal
        norm_path = os.path.normpath(path)
        
        # Ensure the path doesn't start with / or contain ..
        if norm_path.startswith('/') or norm_path.startswith('\\') or '..' in norm_path.split(os.sep):
            return {"error": f"Invalid path: {path}. Path must be relative to the codebase."}
        
        target_path = os.path.join(self.codebase_path, norm_path)
        
        # Ensure the target path is within the codebase
        if not os.path.abspath(target_path).startswith(os.path.abspath(self.codebase_path)):
            return {"error": f"Path traversal attempt detected: {path}. Path must be within the codebase."}
        
        if not os.path.exists(target_path):
            return {"error": f"File does not exist: {path}"}
        
        if not os.path.isfile(target_path):
            return {"error": f"Path is not a file: {path}"}
        
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read(max_length) if max_length else f.read()
            
            return {
                "content": content,
                "path": path,
                "size": len(content),
                "truncated": max_length is not None and len(content) >= max_length
            }
        except Exception as e:
            return {"error": f"Error reading file: {str(e)}"}


class SearchCodebaseTool(BaseTool):
    """Tool for searching for patterns in the codebase."""
    
    name: ClassVar[str] = "search_codebase"
    description: ClassVar[str] = "Searches for patterns in the codebase files."
    codebase_path: str
    
    def __init__(self, codebase_path: str):
        """
        Initialize the SearchCodebaseTool.
        
        Args:
            codebase_path: Absolute path to the codebase
        """
        super().__init__(codebase_path=codebase_path)
        self.codebase_path = codebase_path
    
    def _run(
        self, 
        pattern: str, 
        path: str = ".", 
        file_extensions: List[str] = ["*"],
        exclude_dirs: List[str] = [".git", "node_modules", "venv", "__pycache__"],
        max_results: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for patterns in the codebase.
        
        Args:
            pattern: Pattern to search for
            path: Relative path within the codebase to search in
            file_extensions: List of file extensions to search
            exclude_dirs: Directories to exclude from the search
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        # Normalize the path to prevent path traversal
        norm_path = os.path.normpath(path)
        
        # Ensure the path doesn't start with / or contain ..
        if norm_path.startswith('/') or norm_path.startswith('\\') or '..' in norm_path.split(os.sep):
            return {"error": f"Invalid path: {path}. Path must be relative to the codebase."}
        
        search_path = os.path.join(self.codebase_path, norm_path)
        
        # Ensure the search path is within the codebase
        if not os.path.abspath(search_path).startswith(os.path.abspath(self.codebase_path)):
            return {"error": f"Path traversal attempt detected: {path}. Path must be within the codebase."}
        
        if not os.path.exists(search_path):
            return {"error": f"Path does not exist: {path}"}
        
        if not os.path.isdir(search_path):
            return {"error": f"Path is not a directory: {path}"}
        
        results = []
        
        # Build the list of files to search
        files_to_search = []
        for ext in file_extensions:
            glob_pattern = f"**/*.{ext}" if ext != "*" else "**/*"
            files_to_search.extend(glob.glob(
                os.path.join(search_path, glob_pattern),
                recursive=True
            ))
        
        # Filter out excluded directories
        exclude_paths = [os.path.join(self.codebase_path, d) for d in exclude_dirs]
        files_to_search = [
            f for f in files_to_search 
            if os.path.isfile(f) and not any(
                os.path.commonpath([f, exclude]) == exclude 
                for exclude in exclude_paths if os.path.exists(exclude)
            )
        ]
        
        # Search for the pattern in each file
        for file_path in files_to_search:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        if pattern.lower() in line.lower():
                            rel_path = os.path.relpath(file_path, self.codebase_path)
                            results.append({
                                "file": rel_path,
                                "line": i,
                                "content": line.strip()
                            })
            except Exception:
                # Skip files that can't be read as text
                continue
        
        return {"matches": results[:max_results]}
