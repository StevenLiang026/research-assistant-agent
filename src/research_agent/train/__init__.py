"""Training data assembly (per-step prompt-completion samples) and the per-run manifest.

The trainer itself is scripts/train.py; the export and serving launchers are scripts/export_adapter.py and
scripts/serve_local.py. The owner runs them and reads the curves; this package makes sure every run records
what the plan requires.
"""
