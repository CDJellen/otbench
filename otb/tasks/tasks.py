import os
import json
from typing import List, Optional


class Tasks(object):
    """A singleton helper for recording supported benchmarking tasks."""

    def __init__(self, root_dir: Optional[str] = None) -> None:
        """Read the currently-supported benchmarking task for loaders and evaluators."""
        if root_dir is None:
            root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # obtain root path
        tasks_path = os.path.join(root_dir, 'config', 'tasks.json')
        tasks = json.load(open(tasks_path, 'rb'))

        self.tasks = tasks
        self._build_task_names()
    
    def get_task(self, task_name: str) -> dict:
        """Get a task by name."""
        if self._is_supported_task(task_name=task_name):
            return self._get_task(key=task_name)
        else: raise NotImplementedError

    def list_tasks(self) -> List[str]:
        """List all currently supported tasks."""
        return self.task_names

    def _get_task(self, key: str) -> dict:
        """Traverse a task key, returning bottom-level data."""
        path = key.split(".")
        d = self.tasks.copy()
        for i in range(len(path)):
            d = d[path[i]]
        return d

    def _is_supported_task(self, task_name: str) -> bool:
        """Check whether a task is supported by ot-benchmark."""
        return task_name in self.task_names
    
    def _build_task_names(self) -> None:
        """Gather `.` separated task names for easier inclusion checking."""
        task_names = set()
        task_paths = [(t, t, v) for t, v in self.tasks.items()]
        next_paths = []

        while task_paths or next_paths:
            if not task_paths:
                task_paths = next_paths
                next_paths = []
            
            _, full_k, v = task_paths.pop()

            if 'description' in v:
                task_names.add(full_k)
            else:
                next_paths.extend([(nk, f'{full_k}.{nk}', nv) for nk, nv in v.items()])

        self.task_names = task_names
