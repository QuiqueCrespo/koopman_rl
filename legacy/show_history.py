#!/usr/bin/env python3
import sys
import json
from pathlib import Path

JSONL = Path.home() / ".claude/projects/-home-jq23948-koopman-rl/5ab25a63-4d34-42a3-b09f-3bd8215b36b8.jsonl"

with open(JSONL) as f:
    for line in f:
        entry = json.loads(line)
        # messages are nested under entry["message"]
        msg = entry.get("message", {})
        role = msg.get("role", "")
        if not role:
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        parts.append(c.get("text", ""))
                    elif c.get("type") == "tool_use":
                        inp = c.get("input", {})
                        parts.append(f"[tool: {c.get('name','')} | {json.dumps(inp)[:300]}]")
                    elif c.get("type") == "tool_result":
                        rc = c.get("content", "")
                        if isinstance(rc, list):
                            rc = " ".join(x.get("text","") for x in rc if isinstance(x,dict))
                        parts.append(f"[tool_result: {str(rc)[:500]}]")
            content = " ".join(parts)
        label = {"user": "USER", "assistant": "ASSISTANT"}.get(role, role.upper())
        print(f"{'='*60}")
        print(f"[{label}]")
        print(content)
        print()
