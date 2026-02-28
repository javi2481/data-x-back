import json
import os
from pathlib import Path
from typing import Dict, Any

class RulesLoader:
    def __init__(self, rules_path: str = "rules.json"):
        # Resolve path relative to the project root (assuming engine/ is one level down)
        base_dir = Path(__file__).parent.parent
        self.rules_path = base_dir / rules_path
        self._rules: Dict[str, Any] = {}
        self.load_rules()

    def load_rules(self) -> None:
        """Loads rules from the JSON file into memory."""
        try:
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                self._rules = json.load(f)
        except FileNotFoundError:
            # Fallback to sensible defaults if file is missing during dev
            self._rules = {
                "data_safety": {"mask_pii": True, "allow_original_mutation": False},
                "analysis_policy": {"preferred_library": "pandas"},
                "code_generation_policy": {"allowed_libraries": ["pandas", "numpy"]},
                "output_formatting_policy": {"report_format": "markdown"}
            }
            print(f"Warning: {self.rules_path} not found. Operating with default fallback rules.")

    def get_rules(self) -> Dict[str, Any]:
        """Returns the complete rules dictionary."""
        return self._rules

    def get_safety_policy(self) -> Dict[str, Any]:
        return self._rules.get("data_safety", {})

    def get_analysis_policy(self) -> Dict[str, Any]:
        return self._rules.get("analysis_policy", {})

    def get_code_policy(self) -> Dict[str, Any]:
        return self._rules.get("code_generation_policy", {})
        
    def get_output_policy(self) -> Dict[str, Any]:
        return self._rules.get("output_formatting_policy", {})

# Singleton instance for easy import across the engine
rules_loader = RulesLoader()
