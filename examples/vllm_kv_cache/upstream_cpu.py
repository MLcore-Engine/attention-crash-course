"""Load the pinned upstream CPU block classes without importing the GPU package.

Class and function bodies come directly from upstream/source. Only imports and
module-level type aliases are supplied here. Events stay disabled in the demo.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import pickle
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, NewType, Optional, Union


SOURCE = Path(__file__).parent / "upstream" / "source"


def load_upstream_cpu() -> ModuleType:
    module = ModuleType("_vllm_kv_study_cpu")
    sys.modules[module.__name__] = module
    module.__dict__.update(
        dataclass=dataclass,
        Iterable=Iterable,
        Any=Any,
        Callable=Callable,
        Optional=Optional,
        Union=Union,
        BlockHash=NewType("BlockHash", bytes),
        BlockHashWithGroupId=NewType("BlockHashWithGroupId", bytes),
        NONE_HASH=bytes(32),
        hashlib=hashlib,
        pickle=pickle,
        logger=logging.getLogger(module.__name__),
    )
    selections = {
        "vllm/utils/__init__.py": {"sha256"},
        "vllm/v1/core/kv_cache_utils.py": {
            "make_block_hash_with_group_id",
            "hash_block_tokens",
            "KVCacheBlock",
            "FreeKVCacheBlockQueue",
        },
        "vllm/v1/core/block_pool.py": {"BlockHashToBlockMap", "BlockPool"},
    }
    for relative_path, names in selections.items():
        path = SOURCE / relative_path
        tree = ast.parse(path.read_text(), filename=str(path))
        definitions = [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in names
        ]
        found = {node.name for node in definitions}
        if found != names:
            raise ValueError(f"Missing upstream definitions in {path}: {names - found}")
        future = ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        )
        isolated = ast.Module(body=[future, *definitions], type_ignores=[])
        ast.fix_missing_locations(isolated)
        exec(compile(isolated, str(path), "exec"), module.__dict__)
    return module
