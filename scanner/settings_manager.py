from __future__ import annotations

import json
from pathlib import Path
from typing import Any

class SettingsManager:
    def __init__(self) -> None:
        self.app_dir = Path.home() / ".film_scanner_app"
        self.app_dir.mkdir(parents=True, exist_ok=True)

        self.settings_path = self.app_dir / "settings.json"
        self.default_path = Path(__file__).resolve().parent.parent / "config" / "default_settings.json"
        self._settings = self._load_settings()

    def _load_settings(self) -> dict[str, Any]:
        defaults = {}
        if self.default_path.exists():
            defaults = json.loads(self.default_path.read_text(encoding="utf-8"))

        if self.settings_path.exists():
            try:
                saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
                defaults.update(saved)
            except Exception:
                pass

        return defaults

    def save(self) -> None:
        self.settings_path.write_text(json.dumps(self._settings, indent=2), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        return self._settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._settings[key] = value