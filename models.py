from datetime import datetime
from typing import List, Optional
import json
import os

class ProcessingResult:
    """Simple file-based storage for processing results"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save_result(self, filename: str, result: dict) -> str:
        """Save processing result to file"""
        timestamp = datetime.now().isoformat()
        result_id = f"{timestamp}_{filename}".replace(":", "-").replace(".", "_")
        
        result_data = {
            "id": result_id,
            "timestamp": timestamp,
            "filename": filename,
            "result": result
        }
        
        filepath = os.path.join(self.data_dir, f"{result_id}.json")
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        return result_id
    
    def get_result(self, result_id: str) -> Optional[dict]:
        """Retrieve processing result by ID"""
        filepath = os.path.join(self.data_dir, f"{result_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    
    def list_results(self, limit: int = 50) -> List[dict]:
        """List recent processing results"""
        results = []
        for filename in sorted(os.listdir(self.data_dir), reverse=True)[:limit]:
            if filename.endswith('.json'):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'r') as f:
                    results.append(json.load(f))
        return results