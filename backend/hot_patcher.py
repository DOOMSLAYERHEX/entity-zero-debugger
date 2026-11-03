"""
Hot Patcher & Rollback Manager
Atomically writes patches to disk to trigger Hot Module Reloading (HMR).
Maintains an in-memory rollback stack for immediate one-click reverting.
"""

import os
import time
from typing import Dict, Any, Optional

class HotPatcher:
    def __init__(self):
        # Stack of backups: {patch_id: {"file_path": str, "previous_content": str, "timestamp": float}}
        self.rollback_history = []

    def apply_patch(
        self,
        file_path: str,
        original_snippet: str,
        patched_snippet: str,
        patch_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Applies snippet replacement to file_path and logs backup.
        """
        if not os.path.isfile(file_path):
            return {"success": False, "error": f"File not found: {file_path}"}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if original_snippet not in content:
                # Soft failure: snippet not found exactly
                return {
                    "success": False,
                    "error": "Original code snippet not found in file. It may have already been modified."
                }

            # Backup
            pid = patch_id or f"patch_{int(time.time() * 1000)}"
            self.rollback_history.append({
                "patch_id": pid,
                "file_path": file_path,
                "previous_content": content,
                "timestamp": time.time()
            })

            new_content = content.replace(original_snippet, patched_snippet, 1)

            # Atomic write
            temp_path = f"{file_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            os.replace(temp_path, file_path)

            return {
                "success": True,
                "patch_id": pid,
                "file_path": file_path,
                "bytes_written": len(new_content.encode("utf-8")),
                "timestamp": time.time()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def rollback(self, patch_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Rolls back the most recent patch or specified patch_id.
        """
        if not self.rollback_history:
            return {"success": False, "error": "No patch history to rollback."}

        entry = None
        if patch_id:
            for idx, item in enumerate(self.rollback_history):
                if item["patch_id"] == patch_id:
                    entry = self.rollback_history.pop(idx)
                    break
        else:
            entry = self.rollback_history.pop()

        if not entry:
            return {"success": False, "error": f"Patch {patch_id} not found."}

        try:
            file_path = entry["file_path"]
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(entry["previous_content"])

            return {
                "success": True,
                "rolled_back_id": entry["patch_id"],
                "file_path": file_path
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

