# notes/register_loader_decorator.py
# Test 5: Confirms register_loader uses a Literal type — 'OCTRON' must be
# added to that Literal in movement/io/load.py when the loader is merged.

from movement.io.load import register_loader, load_dataset
import inspect

print("=== register_loader signature ===")
print(inspect.signature(register_loader))

# The Literal annotation lists every supported source_software string.
# 'OCTRON' is not in it yet. Adding the OCTRON loader requires updating this
print("\n=== movement load_dataset source (first 600 chars) ===")
print(inspect.getsource(load_dataset)[:600])