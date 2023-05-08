import os
import json
from typing import Optional


class Tasks(object):
    """A singleton helper for recording supported benchmarking tasks."""

    def __init__(self, root_dir: Optional[str] = None) -> None:
        """Read the currently-supported benchmarking task for loaders and evaluators."""
        if root_dir is None:
            root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # obtain root path
        tasks_path = os.path.join(root_dir, 'tasks', 'tasks.json')
        tasks = json.load(open(tasks_path, 'rb'))

        self.tasks = tasks
        self._build_task_names()

    def is_supported_task(self, task_name: str) -> bool:
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
