"""
Security Tools

This module provides tools for performing security audits on codebases.
"""

import os
import re
from typing import Dict, List, Any, ClassVar

from langchain.tools import BaseTool

class SecurityAuditTool(BaseTool):
    """Tool for performing security audits on a codebase."""
    
    name: ClassVar[str] = "security_audit"
    description: ClassVar[str] = "Performs a security audit on a codebase, identifying potential security risks and vulnerabilities."
    codebase_path: str
    
    def __init__(self, codebase_path: str):
        """
        Initialize the SecurityAuditTool.
        
        Args:
            codebase_path: Absolute path to the codebase
        """
        super().__init__(codebase_path=codebase_path)
        self.codebase_path = codebase_path
    
    def _run(self, path: str = ".", scan_depth: str = "medium") -> Dict[str, Any]:
        """
        Perform a security audit on a codebase.
        
        Args:
            path: Relative path within the codebase to audit
            scan_depth: Depth of the scan (low, medium, high)
            
        Returns:
            Dictionary with security audit results
        """
        target_path = os.path.join(self.codebase_path, path)
        
        if not os.path.exists(target_path):
            return {"error": f"Path does not exist: {path}"}
        
        # Check for common security issues
        issues = []
        
        # Check for hardcoded secrets
        secrets = self._check_for_hardcoded_secrets(target_path)
        issues.extend(secrets)
        
        # Check for insecure dependencies
        dependency_issues = self._check_dependencies(target_path)
        issues.extend(dependency_issues)
        
        # Check for common vulnerabilities
        vulnerability_issues = self._check_for_vulnerabilities(target_path, scan_depth)
        issues.extend(vulnerability_issues)
        
        # Categorize issues by severity
        categorized_issues = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }
        
        for issue in issues:
            categorized_issues[issue["severity"]].append(issue)
        
        return {
            "issues": issues,
            "categorized_issues": categorized_issues,
            "summary": {
                "total_issues": len(issues),
                "critical": len(categorized_issues["critical"]),
                "high": len(categorized_issues["high"]),
                "medium": len(categorized_issues["medium"]),
                "low": len(categorized_issues["low"])
            }
        }
    
    def _check_for_hardcoded_secrets(self, path: str) -> List[Dict[str, Any]]:
        """Check for hardcoded secrets in the codebase."""
        issues = []
        
        # Patterns for common secrets
        secret_patterns = {
            "API Key": [
                r"api[_-]?key[\"']?\s*[:=]\s*[\"']([a-zA-Z0-9]{16,})[\"']",
                r"apikey[\"']?\s*[:=]\s*[\"']([a-zA-Z0-9]{16,})[\"']"
            ],
            "AWS Key": [
                r"AKIA[0-9A-Z]{16}",
                r"aws[_-]?access[_-]?key[_-]?id[\"']?\s*[:=]\s*[\"']([A-Za-z0-9/+=]{16,})[\"']"
            ],
            "Password": [
                r"password[\"']?\s*[:=]\s*[\"']([^\"']{8,})[\"']",
                r"passwd[\"']?\s*[:=]\s*[\"']([^\"']{8,})[\"']",
                r"pwd[\"']?\s*[:=]\s*[\"']([^\"']{8,})[\"']"
            ],
            "Private Key": [
                r"-----BEGIN [A-Z]+ PRIVATE KEY-----"
            ],
            "Token": [
                r"token[\"']?\s*[:=]\s*[\"']([a-zA-Z0-9]{16,})[\"']",
                r"auth[_-]?token[\"']?\s*[:=]\s*[\"']([a-zA-Z0-9]{16,})[\"']"
            ]
        }
        
        # Find all code files
        code_files = []
        for root, _, files in os.walk(path):
            # Skip hidden directories and common non-code directories
            if any(part.startswith(".") for part in root.split(os.sep)) and ".github" not in root:
                continue
            if any(part in ["node_modules", "venv", "__pycache__"] for part in root.split(os.sep)):
                continue
            
            for file in files:
                if file.endswith((".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".rb", ".go", ".php", ".conf", ".yml", ".yaml", ".json", ".xml", ".env")):
                    code_files.append(os.path.join(root, file))
        
        # Check each file for secrets
        for file_path in code_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                rel_path = os.path.relpath(file_path, self.codebase_path)
                
                for secret_type, patterns in secret_patterns.items():
                    for pattern in patterns:
                        for match in re.finditer(pattern, content):
                            # Don't include the actual secret in the report
                            issues.append({
                                "type": "hardcoded_secret",
                                "secret_type": secret_type,
                                "file": rel_path,
                                "line": content.count("\n", 0, match.start()) + 1,
                                "severity": "critical",
                                "description": f"Hardcoded {secret_type} found in {rel_path}"
                            })
            except Exception:
                # Skip files that can't be read
                continue
        
        return issues
    
    def _check_dependencies(self, path: str) -> List[Dict[str, Any]]:
        """Check for insecure dependencies in the codebase."""
        issues = []
        
        # Check package.json for Node.js projects
        package_json_path = os.path.join(path, "package.json")
        if os.path.exists(package_json_path):
            try:
                import json
                with open(package_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Example vulnerable packages (in a real implementation, this would be a more comprehensive list)
                vulnerable_packages = {
                    "lodash": ["<4.17.21", "high", "Prototype Pollution"],
                    "express": ["<4.17.3", "medium", "Open Redirect"],
                    "node-fetch": ["<2.6.7", "high", "SSRF"],
                    "minimist": ["<1.2.6", "medium", "Prototype Pollution"]
                }
                
                dependencies = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                
                for package, version in dependencies.items():
                    if package in vulnerable_packages:
                        version_constraint, severity, vulnerability = vulnerable_packages[package]
                        # This is a simplified version check, in a real implementation we would use semver
                        if version.startswith("<") or version.startswith("^") or version.startswith("~"):
                            issues.append({
                                "type": "vulnerable_dependency",
                                "package": package,
                                "version": version,
                                "recommendation": f"Update to {package} {version_constraint.replace('<', '>=')}",
                                "severity": severity,
                                "vulnerability": vulnerability,
                                "file": "package.json",
                                "description": f"Potentially vulnerable dependency: {package}@{version} ({vulnerability})"
                            })
            except Exception:
                pass
        
        # Check requirements.txt for Python projects
        requirements_path = os.path.join(path, "requirements.txt")
        if os.path.exists(requirements_path):
            try:
                with open(requirements_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Example vulnerable packages (in a real implementation, this would be a more comprehensive list)
                vulnerable_packages = {
                    "django": ["<3.2.14", "high", "SQL Injection"],
                    "flask": ["<2.0.2", "medium", "Open Redirect"],
                    "requests": ["<2.27.1", "medium", "CRLF Injection"],
                    "pyyaml": ["<5.4", "high", "Code Execution"]
                }
                
                for line in content.splitlines():
                    if "==" in line:
                        package, version = line.split("==", 1)
                        package = package.strip()
                        version = version.strip()
                        
                        if package in vulnerable_packages:
                            version_constraint, severity, vulnerability = vulnerable_packages[package]
                            # This is a simplified version check, in a real implementation we would use semver
                            issues.append({
                                "type": "vulnerable_dependency",
                                "package": package,
                                "version": version,
                                "recommendation": f"Update to {package} {version_constraint.replace('<', '>=')}",
                                "severity": severity,
                                "vulnerability": vulnerability,
                                "file": "requirements.txt",
                                "description": f"Potentially vulnerable dependency: {package}=={version} ({vulnerability})"
                            })
            except Exception:
                pass
        
        return issues
    
    def _check_for_vulnerabilities(self, path: str, scan_depth: str) -> List[Dict[str, Any]]:
        """Check for common vulnerabilities in the codebase."""
        issues = []
        
        # Define vulnerability patterns based on scan depth
        vulnerability_patterns = {
            "SQL Injection": {
                "patterns": [
                    r"execute\([\"']SELECT .* WHERE .* = ['\"].*\s*\+\s*.*['\"]",
                    r"execute\([\"']INSERT INTO .* VALUES\s*\(.*\s*\+\s*.*\)",
                    r"execute\([\"']UPDATE .* SET .* = .*\s*\+\s*.*['\"]",
                    r"execute\([\"']DELETE FROM .* WHERE .*\s*\+\s*.*['\"]",
                    r"raw\([\"']SELECT .* WHERE .* = ['\"].*\s*\+\s*.*['\"]"
                ],
                "severity": "high",
                "file_extensions": [".py", ".js", ".php", ".java", ".rb"]
            },
            "XSS": {
                "patterns": [
                    r"innerHTML\s*=",
                    r"document\.write\(",
                    r"\.html\(",
                    r"eval\(",
                    r"dangerouslySetInnerHTML"
                ],
                "severity": "high",
                "file_extensions": [".js", ".jsx", ".ts", ".tsx", ".html", ".php"]
            },
            "Command Injection": {
                "patterns": [
                    r"exec\([\"'].*\s*\+\s*.*['\"]",
                    r"spawn\([\"'].*\s*\+\s*.*['\"]",
                    r"system\([\"'].*\s*\+\s*.*['\"]",
                    r"popen\([\"'].*\s*\+\s*.*['\"]",
                    r"subprocess\.call\([\"'].*\s*\+\s*.*['\"]",
                    r"subprocess\.Popen\([\"'].*\s*\+\s*.*['\"]",
                    r"os\.system\([\"'].*\s*\+\s*.*['\"]"
                ],
                "severity": "critical",
                "file_extensions": [".py", ".js", ".php", ".rb", ".go", ".java"]
            },
            "Insecure File Operations": {
                "patterns": [
                    r"open\([\"'].*\s*\+\s*.*['\"]",
                    r"readFile\([\"'].*\s*\+\s*.*['\"]",
                    r"writeFile\([\"'].*\s*\+\s*.*['\"]",
                    r"fs\.read",
                    r"fs\.write"
                ],
                "severity": "medium",
                "file_extensions": [".py", ".js", ".php", ".rb", ".go", ".java"]
            },
            "Insecure Deserialization": {
                "patterns": [
                    r"pickle\.loads",
                    r"yaml\.load\(",
                    r"marshal\.loads",
                    r"json\.parse\(",
                    r"unserialize\("
                ],
                "severity": "high",
                "file_extensions": [".py", ".js", ".php", ".rb", ".java"]
            }
        }
        
        # Adjust scan depth
        if scan_depth == "low":
            # Only check for critical vulnerabilities
            vulnerability_patterns = {k: v for k, v in vulnerability_patterns.items() if v["severity"] == "critical"}
        elif scan_depth == "medium":
            # Check for critical and high vulnerabilities
            vulnerability_patterns = {k: v for k, v in vulnerability_patterns.items() if v["severity"] in ["critical", "high"]}
        
        # Find all code files
        code_files = []
        for root, _, files in os.walk(path):
            # Skip hidden directories and common non-code directories
            if any(part.startswith(".") for part in root.split(os.sep)) and ".github" not in root:
                continue
            if any(part in ["node_modules", "venv", "__pycache__"] for part in root.split(os.sep)):
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)
                code_files.append((file_path, ext))
        
        # Check each file for vulnerabilities
        for file_path, ext in code_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                rel_path = os.path.relpath(file_path, self.codebase_path)
                
                for vuln_type, vuln_info in vulnerability_patterns.items():
                    if ext in vuln_info["file_extensions"]:
                        for pattern in vuln_info["patterns"]:
                            for match in re.finditer(pattern, content):
                                issues.append({
                                    "type": "vulnerability",
                                    "vulnerability_type": vuln_type,
                                    "file": rel_path,
                                    "line": content.count("\n", 0, match.start()) + 1,
                                    "severity": vuln_info["severity"],
                                    "description": f"Potential {vuln_type} vulnerability in {rel_path}",
                                    "code_snippet": match.group(0)
                                })
            except Exception:
                # Skip files that can't be read
                continue
        
        return issues
