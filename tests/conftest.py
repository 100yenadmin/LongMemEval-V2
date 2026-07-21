"""Keep the official source checkout importable during focused pytest runs."""

from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The focused Hermes adapter fixtures do not exercise the optional coding-agent
# backends.  The official module imports every backend eagerly, while this
# provider-free test environment intentionally does not install openai-agents.
# Stub only those sibling modules so the registry contract can be tested
# without pulling a model/provider dependency into this slice.
for module_name, class_name in {
    "memory_modules.codex": "CodexMemory",
    "memory_modules.agentrunbook_c": "AgentRunbookC",
    "memory_modules.agentrunbook_c_v2": "AgentRunbookCV2",
    "memory_modules.agentrunbook_r": "AgentRunbookR",
    "memory_modules.rag": "RagMemory",
}.items():
    module = types.ModuleType(module_name)
    setattr(module, class_name, type(class_name, (), {}))
    sys.modules.setdefault(module_name, module)
