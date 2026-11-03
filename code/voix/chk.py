"""Verifie que whisper et torch sont importables dans le python de l'hote."""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import importlib

for name in ("torch", "whisper", "numpy"):
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "?")
        print(f"{name:8s} present  {version}")
    except ImportError as exc:
        print(f"{name:8s} absent   {exc}")
