"""Model families (docs/m7-家族层设计.md): what differs between the base models a run can train and serve.

core.py holds the family specification (Family) and a family loaded for one model folder (LoadedFamily,
load_family), which renders, encodes and reads text for that model; registry.py the registered families
and how a model folder's family is found; toolformats/ how each family's model writes its tool calls.
"""
