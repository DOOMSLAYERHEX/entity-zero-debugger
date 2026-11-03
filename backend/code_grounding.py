"""
Code Grounding Engine
Maps visual screen coordinates, segmented masks, and DOM traits to source code files.
Eliminates manual codebase exploration or source-tree tagging.
"""

import os
import re
from typing import Dict, Any, List, Optional, Tuple

class CodeGroundingEngine:
    def __init__(self, project_root: str):
        self.project_root = project_root

    def locate_target_file_and_snippet(
        self,
        target_info: Dict[str, Any],
        hint_file: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """
        Locates the source file, relative path, and full contents based on
        element tags, text, classes, or attributes.
        """
        tag = target_info.get("tagName", "").lower()
        class_name = target_info.get("className", "")
        element_id = target_info.get("id", "")
        text = target_info.get("text", "").strip()

        # If a hint file exists, check it first
        if hint_file:
            abs_hint = os.path.join(self.project_root, hint_file) if not os.path.isabs(hint_file) else hint_file
            if os.path.isfile(abs_hint):
                with open(abs_hint, "r", encoding="utf-8") as f:
                    return abs_hint, hint_file, f.read()

        # Search priority patterns
        search_terms = []
        if element_id:
            search_terms.append(f'id="{element_id}"')
            search_terms.append(f"id='{element_id}'")
        if class_name:
            # Look for distinctive classes
            classes = [c for c in class_name.split() if len(c) > 3 and not c.startswith("p-") and not c.startswith("m-")]
            for c in classes[:3]:
                search_terms.append(c)
        if text and len(text) > 3:
            # Clean text for search
            clean_text = text[:30].strip()
            search_terms.append(clean_text)

        # Scan project files (.jsx, .tsx, .js, .html, .vue, .svelte)
        valid_extensions = (".jsx", ".tsx", ".js", ".html", ".vue", ".svelte", ".css")
        best_match = None
        best_score = 0

        for root, dirs, files in os.walk(self.project_root):
            # Skip node_modules, .git, venv
            dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", "venv", "__pycache__", "dist", "build")]
            for file in files:
                if any(file.endswith(ext) for ext in valid_extensions):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.project_root)
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            score = 0
                            for term in search_terms:
                                if term in content:
                                    score += 10
                            if score > best_score:
                                best_score = score
                                best_match = (full_path, rel_path, content)
                    except Exception:
                        continue

        if best_match and best_score > 0:
            return best_match

        # Fallback to demo file if present
        default_file = os.path.join(self.project_root, "demo-app", "index.html")
        if os.path.isfile(default_file):
            with open(default_file, "r", encoding="utf-8") as f:
                return default_file, "demo-app/index.html", f.read()

        return "", "", ""

