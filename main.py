from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY environment variable.")
    return Groq(api_key=api_key)


def generate_wbs(scope_text: str) -> list[dict[str, Any]]:
    prompt = f"""
You are an expert project manager.

Convert the given scope into a STRICT Work Breakdown Structure (WBS).

RULES:
- 4 levels only
- Format codes like 1, 1.1, 1.1.1
- Max 5 Level 1 items
- Output ONLY valid JSON

FORMAT:
[
  {{"level": 1, "code": "1", "name": "Phase"}},
  {{"level": 2, "code": "1.1", "name": "Sub-phase"}},
  {{"level": 3, "code": "1.1.1", "name": "Task"}},
  {{"level": 4, "code": "1.1.1.1", "name": "Sub-Tasks"}}
]

SCOPE:
{scope_text[:8000]}
"""

    client = get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content or ""

    try:
        parsed = json.loads(content)
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON list.")
        return parsed
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM did not return valid JSON: {exc}") from exc


def build_tree(wbs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    root_nodes: list[dict[str, Any]] = []

    for item in wbs:
        code = str(item.get("code", "")).strip()
        name = str(item.get("name", "")).strip()
        level = int(item.get("level", 0))
        if not code or not name or level <= 0:
            continue

        node = {
            "id": code,
            "label": name,
            "level": level,
            "description": "",
            "children": [],
        }
        nodes[code] = node

    for code, node in nodes.items():
        if "." in code:
            parent_code = ".".join(code.split(".")[:-1])
            parent = nodes.get(parent_code)
            if parent:
                parent["children"].append(node)
            else:
                root_nodes.append(node)
        else:
            root_nodes.append(node)

    return root_nodes


@app.post("/generate-mermaid-chart")
async def generate_wbs_from_upload(text : str):
    
    scope_text = text
    if not scope_text:
        raise HTTPException(status_code=400, detail="Uploaded document has no readable text.")

    wbs = generate_wbs(scope_text)
    nodes = build_tree(wbs)
    return {
        "scopeText": scope_text,
        "wbs": wbs,
        "nodes": nodes,
    }
