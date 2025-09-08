"""
Model Registry Manager for versioning and tracking trained models
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

class ModelRegistry:
    def __init__(self, registry_path: str = "versioning/model_registry.json"):
        """
        Initialize model registry
        
        Args:
            registry_path: Path to the registry JSON file
        """
        self.registry_path = registry_path
        self.registry = self.load_registry()
    
    def load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load registry from JSON file"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading registry: {e}")
                return {"sign_language": {}, "braille": {}}
        else:
            return {"sign_language": {}, "braille": {}}
    
    def save_registry(self) -> bool:
        """Save registry to JSON file"""
        try:
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
            return True
        except IOError as e:
            print(f"Error saving registry: {e}")
            return False
    
    def register_model(self, model_type: str, version: str, model_path: str, 
                      accuracy: float, metadata: Optional[Dict] = None) -> bool:
        """
        Register a new model version
        
        Args:
            model_type: Type of model ('sign_language' or 'braille')
            version: Version identifier (e.g., 'v1', 'v2')
            model_path: Path to the model file
            accuracy: Model accuracy
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        if model_type not in self.registry:
            self.registry[model_type] = {}
        
        model_info = {
            "path": model_path,
            "accuracy": accuracy,
            "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata or {}
        }
        
        self.registry[model_type][version] = model_info
        
        return self.save_registry()
    
    def get_model_info(self, model_type: str, version: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model version"""
        if model_type in self.registry and version in self.registry[model_type]:
            return self.registry[model_type][version]
        return None
    
    def list_models(self, model_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        List all models or models of a specific type
        
        Args:
            model_type: Optional model type filter
            
        Returns:
            Dictionary of models
        """
        if model_type:
            return self.registry.get(model_type, {})
        return self.registry
    
    def get_latest_version(self, model_type: str) -> Optional[str]:
        """Get the latest version of a model type"""
        if model_type not in self.registry:
            return None
        
        versions = list(self.registry[model_type].keys())
        if not versions:
            return None
        
        # Simple version comparison (assumes v1, v2, v3, etc.)
        try:
            version_numbers = [int(v[1:]) for v in versions if v.startswith('v')]
            if version_numbers:
                return f"v{max(version_numbers)}"
        except ValueError:
            pass
        
        # Fallback to lexicographic sorting
        return max(versions)
    
    def delete_model(self, model_type: str, version: str) -> bool:
        """
        Delete a model version from registry
        
        Args:
            model_type: Type of model
            version: Version to delete
            
        Returns:
            True if successful, False otherwise
        """
        if model_type in self.registry and version in self.registry[model_type]:
            del self.registry[model_type][version]
            return self.save_registry()
        return False
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about all models"""
        stats = {
            "total_models": 0,
            "by_type": {},
            "latest_versions": {}
        }
        
        for model_type, models in self.registry.items():
            stats["by_type"][model_type] = len(models)
            stats["total_models"] += len(models)
            stats["latest_versions"][model_type] = self.get_latest_version(model_type)
        
        return stats
