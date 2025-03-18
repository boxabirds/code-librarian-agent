"""
Code Analysis Tools

This module provides tools for analyzing code structure, architecture, and generating diagrams.
"""

import os
import re
import json
from typing import Dict, List, Any, Optional, ClassVar

from langchain.tools import BaseTool

class AnalyzeCodeTool(BaseTool):
    """Tool for analyzing the structure and architecture of a codebase."""
    
    name: ClassVar[str] = "analyze_code"
    description: ClassVar[str] = "Analyzes the structure and architecture of a codebase, identifying key components, patterns, and relationships."
    codebase_path: str
    
    def __init__(self, codebase_path: str):
        """
        Initialize the AnalyzeCodeTool.
        
        Args:
            codebase_path: Absolute path to the codebase
        """
        super().__init__(codebase_path=codebase_path)
        self.codebase_path = codebase_path
    
    def _run(self, path: str = ".", depth: int = 3) -> Dict[str, Any]:
        """
        Analyze the structure and architecture of a codebase.
        
        Args:
            path: Relative path within the codebase to analyze
            depth: Maximum depth to analyze directory structure
            
        Returns:
            Dictionary with analysis results
        """
        target_path = os.path.join(self.codebase_path, path)
        
        if not os.path.exists(target_path):
            return {"error": f"Path does not exist: {path}"}
        
        # Analyze directory structure
        structure = self._analyze_directory_structure(target_path, depth)
        
        # Identify key files
        key_files = self._identify_key_files(target_path)
        
        # Identify programming languages
        languages = self._identify_languages(target_path)
        
        # Identify frameworks and libraries
        frameworks = self._identify_frameworks(target_path)
        
        # Identify architecture patterns
        patterns = self._identify_architecture_patterns(target_path)
        
        return {
            "structure": structure,
            "key_files": key_files,
            "languages": languages,
            "frameworks": frameworks,
            "patterns": patterns
        }
    
    def _analyze_directory_structure(self, path: str, max_depth: int, current_depth: int = 0) -> Dict[str, Any]:
        """Analyze the directory structure of the codebase."""
        if current_depth > max_depth:
            return {"name": os.path.basename(path), "type": "directory", "truncated": True}
        
        result = {"name": os.path.basename(path), "type": "directory", "children": []}
        
        if os.path.isdir(path):
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                
                # Skip hidden files and common non-code directories
                if item.startswith(".") or item in ["node_modules", "venv", "__pycache__", ".git"]:
                    continue
                
                if os.path.isdir(item_path):
                    result["children"].append(
                        self._analyze_directory_structure(item_path, max_depth, current_depth + 1)
                    )
                else:
                    result["children"].append({"name": item, "type": "file"})
        
        return result
    
    def _identify_key_files(self, path: str) -> List[Dict[str, str]]:
        """Identify key files in the codebase."""
        key_files = []
        
        # Common key files to look for
        key_file_patterns = [
            # Configuration files
            "package.json", "requirements.txt", "setup.py", "Dockerfile", "docker-compose.yml",
            "Makefile", ".env", "config.py", "settings.py", "pyproject.toml", "poetry.lock",
            
            # Documentation
            "README.md", "CONTRIBUTING.md", "CHANGELOG.md", "LICENSE",
            
            # Entry points
            "main.py", "app.py", "index.js", "server.js", "manage.py",
            
            # CI/CD
            ".github/workflows/*.yml", ".gitlab-ci.yml", "Jenkinsfile", ".travis.yml"
        ]
        
        for root, dirs, files in os.walk(path):
            # Skip hidden directories and common non-code directories
            if any(part.startswith(".") for part in root.split(os.sep)) and ".github" not in root:
                continue
            if any(part in ["node_modules", "venv", "__pycache__"] for part in root.split(os.sep)):
                continue
            
            for pattern in key_file_patterns:
                if "*" in pattern:
                    # Handle wildcard patterns
                    pattern_dir = os.path.dirname(pattern)
                    pattern_file = os.path.basename(pattern)
                    
                    if pattern_dir and not root.endswith(pattern_dir) and pattern_dir not in root:
                        continue
                    
                    for file in files:
                        if self._match_wildcard(file, pattern_file):
                            rel_path = os.path.relpath(os.path.join(root, file), self.codebase_path)
                            key_files.append({"path": rel_path, "type": "configuration"})
                else:
                    # Handle exact matches
                    if pattern in files:
                        rel_path = os.path.relpath(os.path.join(root, pattern), self.codebase_path)
                        key_files.append({"path": rel_path, "type": self._determine_file_type(pattern)})
        
        return key_files
    
    def _match_wildcard(self, filename: str, pattern: str) -> bool:
        """Match a filename against a wildcard pattern."""
        pattern = pattern.replace(".", "\\.").replace("*", ".*")
        return re.match(f"^{pattern}$", filename) is not None
    
    def _determine_file_type(self, filename: str) -> str:
        """Determine the type of a file based on its name."""
        if filename in ["README.md", "CONTRIBUTING.md", "CHANGELOG.md", "LICENSE"]:
            return "documentation"
        elif filename in ["package.json", "requirements.txt", "setup.py", "pyproject.toml", "poetry.lock"]:
            return "dependencies"
        elif filename in ["Dockerfile", "docker-compose.yml"]:
            return "deployment"
        elif filename in [".github/workflows/*.yml", ".gitlab-ci.yml", "Jenkinsfile", ".travis.yml"]:
            return "ci_cd"
        elif filename in ["main.py", "app.py", "index.js", "server.js", "manage.py"]:
            return "entry_point"
        else:
            return "configuration"
    
    def _identify_languages(self, path: str) -> Dict[str, int]:
        """Identify programming languages used in the codebase."""
        languages = {}
        
        # File extension to language mapping
        extension_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".jsx": "React",
            ".tsx": "React (TypeScript)",
            ".html": "HTML",
            ".css": "CSS",
            ".scss": "SCSS",
            ".java": "Java",
            ".rb": "Ruby",
            ".go": "Go",
            ".rs": "Rust",
            ".php": "PHP",
            ".c": "C",
            ".cpp": "C++",
            ".cs": "C#",
            ".swift": "Swift",
            ".kt": "Kotlin",
            ".sh": "Shell",
            ".md": "Markdown",
            ".json": "JSON",
            ".yml": "YAML",
            ".yaml": "YAML",
            ".xml": "XML",
            ".sql": "SQL"
        }
        
        for root, _, files in os.walk(path):
            # Skip hidden directories and common non-code directories
            if any(part.startswith(".") for part in root.split(os.sep)) and ".github" not in root:
                continue
            if any(part in ["node_modules", "venv", "__pycache__"] for part in root.split(os.sep)):
                continue
            
            for file in files:
                _, ext = os.path.splitext(file)
                if ext in extension_map:
                    language = extension_map[ext]
                    languages[language] = languages.get(language, 0) + 1
        
        return languages
    
    def _identify_frameworks(self, path: str) -> List[str]:
        """Identify frameworks and libraries used in the codebase."""
        frameworks = set()
        
        # Check for package.json (Node.js)
        package_json_path = os.path.join(path, "package.json")
        if os.path.exists(package_json_path):
            try:
                with open(package_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                dependencies = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                
                # Map common npm packages to frameworks
                framework_map = {
                    "react": "React",
                    "vue": "Vue.js",
                    "angular": "Angular",
                    "next": "Next.js",
                    "nuxt": "Nuxt.js",
                    "express": "Express.js",
                    "koa": "Koa.js",
                    "nest": "NestJS",
                    "electron": "Electron",
                    "react-native": "React Native"
                }
                
                for dep in dependencies:
                    for key, value in framework_map.items():
                        if key in dep.lower():
                            frameworks.add(value)
            except Exception:
                pass
        
        # Check for requirements.txt (Python)
        requirements_path = os.path.join(path, "requirements.txt")
        if os.path.exists(requirements_path):
            try:
                with open(requirements_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Map common Python packages to frameworks
                framework_map = {
                    "django": "Django",
                    "flask": "Flask",
                    "fastapi": "FastAPI",
                    "tornado": "Tornado",
                    "pyramid": "Pyramid",
                    "sqlalchemy": "SQLAlchemy",
                    "tensorflow": "TensorFlow",
                    "pytorch": "PyTorch",
                    "pandas": "Pandas",
                    "numpy": "NumPy",
                    "scikit-learn": "scikit-learn"
                }
                
                for key, value in framework_map.items():
                    if re.search(rf"{key}[=~<>]", content, re.IGNORECASE):
                        frameworks.add(value)
            except Exception:
                pass
        
        # Check for specific framework files
        framework_files = {
            "manage.py": "Django",
            "app/models.py": "Django",
            "wsgi.py": "WSGI",
            "asgi.py": "ASGI",
            "Gemfile": "Ruby on Rails",
            "build.gradle": "Android/Gradle",
            "pom.xml": "Java/Maven",
            "AndroidManifest.xml": "Android",
            "Info.plist": "iOS",
            "CMakeLists.txt": "CMake",
            "Cargo.toml": "Rust/Cargo"
        }
        
        for file_path, framework in framework_files.items():
            if os.path.exists(os.path.join(path, file_path)):
                frameworks.add(framework)
        
        return list(frameworks)
    
    def _identify_architecture_patterns(self, path: str) -> List[str]:
        """Identify architecture patterns used in the codebase."""
        patterns = set()
        
        # Check directory structure for common patterns
        directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        
        # MVC pattern
        if all(d in directories for d in ["models", "views", "controllers"]):
            patterns.add("Model-View-Controller (MVC)")
        
        # MVVM pattern
        if all(d in directories for d in ["models", "views", "viewmodels"]):
            patterns.add("Model-View-ViewModel (MVVM)")
        
        # Clean Architecture
        if all(d in directories for d in ["domain", "data", "presentation"]) or \
           all(d in directories for d in ["entities", "usecases", "interfaces"]):
            patterns.add("Clean Architecture")
        
        # Microservices
        if "services" in directories or "microservices" in directories:
            patterns.add("Microservices")
        
        # Serverless
        if "functions" in directories or "lambdas" in directories:
            patterns.add("Serverless")
        
        # Check for specific files indicating patterns
        if os.path.exists(os.path.join(path, "docker-compose.yml")):
            patterns.add("Containerization")
        
        if os.path.exists(os.path.join(path, ".github/workflows")) or \
           os.path.exists(os.path.join(path, ".gitlab-ci.yml")) or \
           os.path.exists(os.path.join(path, "Jenkinsfile")):
            patterns.add("CI/CD")
        
        return list(patterns)


class GenerateArchitectureDiagramTool(BaseTool):
    """Tool for generating Mermaid diagrams for a codebase."""
    
    name: ClassVar[str] = "generate_architecture_diagram"
    description: ClassVar[str] = "Generates Mermaid diagrams for visualizing different aspects of a codebase."
    codebase_path: str
    
    def __init__(self, codebase_path: str):
        """
        Initialize the GenerateArchitectureDiagramTool.
        
        Args:
            codebase_path: Absolute path to the codebase
        """
        super().__init__(codebase_path=codebase_path)
        self.codebase_path = codebase_path
    
    def _run(self, diagram_type: str, path: str = ".") -> Dict[str, Any]:
        """
        Generate a Mermaid diagram for the codebase.
        
        Args:
            diagram_type: Type of diagram to generate (architecture, class, deployment, cicd)
            path: Relative path within the codebase to analyze
            
        Returns:
            Dictionary with the Mermaid diagram code
        """
        target_path = os.path.join(self.codebase_path, path)
        
        if not os.path.exists(target_path):
            return {"error": f"Path does not exist: {path}"}
        
        if diagram_type == "architecture":
            return {"diagram": self._generate_architecture_diagram(target_path)}
        elif diagram_type == "class":
            return {"diagram": self._generate_class_diagram(target_path)}
        elif diagram_type == "deployment":
            return {"diagram": self._generate_deployment_diagram(target_path)}
        elif diagram_type == "cicd":
            return {"diagram": self._generate_cicd_diagram(target_path)}
        else:
            return {"error": f"Unknown diagram type: {diagram_type}"}
    
    def _generate_architecture_diagram(self, path: str) -> str:
        """Generate a Mermaid architecture diagram for the codebase."""
        # This is a placeholder implementation that would be replaced with actual
        # code analysis in a production implementation. For now, we'll return a
        # template that the LLM can fill in based on its analysis.
        
        return """
```mermaid
graph TD
    %% Architecture Diagram Template
    %% Replace with actual components and relationships
    
    Client[Client] --> FE[Frontend]
    FE --> API[API Layer]
    API --> BL[Business Logic]
    BL --> DB[(Database)]
    
    %% Add more components and relationships based on analysis
```
"""
    
    def _generate_class_diagram(self, path: str) -> str:
        """Generate a Mermaid class diagram for the codebase."""
        # This is a placeholder implementation that would be replaced with actual
        # code analysis in a production implementation. For now, we'll return a
        # template that the LLM can fill in based on its analysis.
        
        return """
```mermaid
classDiagram
    %% Class Diagram Template
    %% Replace with actual classes, properties, and relationships
    
    class Entity {
        +id: string
        +createdAt: datetime
        +updatedAt: datetime
    }
    
    class User {
        +username: string
        +email: string
        +password: string
        +authenticate()
    }
    
    Entity <|-- User
    
    %% Add more classes and relationships based on analysis
```
"""
    
    def _generate_deployment_diagram(self, path: str) -> str:
        """Generate a Mermaid deployment diagram for the codebase."""
        # This is a placeholder implementation that would be replaced with actual
        # code analysis in a production implementation. For now, we'll return a
        # template that the LLM can fill in based on its analysis.
        
        return """
```mermaid
graph TD
    %% Deployment Diagram Template
    %% Replace with actual deployment components
    
    subgraph Cloud
        LB[Load Balancer]
        
        subgraph AppServers
            App1[App Server 1]
            App2[App Server 2]
        end
        
        subgraph Database
            Primary[(Primary DB)]
            Replica[(Replica DB)]
        end
        
        CDN[Content Delivery Network]
    end
    
    Client[Client] --> CDN
    Client --> LB
    LB --> App1
    LB --> App2
    App1 --> Primary
    App2 --> Primary
    Primary --> Replica
    
    %% Add more deployment components based on analysis
```
"""
    
    def _generate_cicd_diagram(self, path: str) -> str:
        """Generate a Mermaid CI/CD diagram for the codebase."""
        # This is a placeholder implementation that would be replaced with actual
        # code analysis in a production implementation. For now, we'll return a
        # template that the LLM can fill in based on its analysis.
        
        return """
```mermaid
graph LR
    %% CI/CD Diagram Template
    %% Replace with actual CI/CD pipeline
    
    Code[Code] --> Build
    Build --> Test
    Test --> Deploy
    
    subgraph CI
        Build
        Test
    end
    
    subgraph CD
        Deploy --> Staging
        Staging --> Production
    end
    
    %% Add more CI/CD steps based on analysis
```
"""
