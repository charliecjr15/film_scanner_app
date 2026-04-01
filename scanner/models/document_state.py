from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from scanner.models.image_job import ImageJob

@dataclass
class DocumentState:
    queue: List[ImageJob] = field(default_factory=list)
    selected_index: int = -1

    @property
    def current_job(self) -> Optional[ImageJob]:
        if 0 <= self.selected_index < len(self.queue):
            return self.queue[self.selected_index]
        return None