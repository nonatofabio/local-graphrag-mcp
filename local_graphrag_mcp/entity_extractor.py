#!/usr/bin/env python3
"""
Entity extraction for GraphRAG knowledge graph construction.

Extracts structured entities (nodes) and relationships (edges) from text
using either:
- Ollama (local, default) with Llama 3.2 3B
- Claude API (cloud, optional) for higher precision
"""

import json
import os
from typing import Any, Optional
from pathlib import Path


EXTRACTION_PROMPT = """You are an expert knowledge graph extractor. Your task is to analyze the following text and extract entities and their relationships in a structured format.

Instructions:
1. Identify key entities (people, organizations, services, databases, concepts, etc.)
2. For each entity, create:
   - A unique ID (format: "entity:<lowercase_name_with_underscores>")
   - A label (entity type: Person, Service, Database, Concept, etc.)
   - A clear description (1-2 sentences explaining what this entity is)
   - Properties as a JSON object (any relevant metadata)

3. Identify relationships between entities
4. For each relationship, create:
   - source: ID of the source entity
   - target: ID of the target entity
   - label: Relationship type (MANAGES, CONNECTS_TO, USES, OWNS, etc.)
   - properties: JSON object with relationship metadata

5. Return ONLY valid JSON with this exact structure:
{
  "nodes": [
    {
      "id": "entity:example",
      "label": "Service",
      "description": "A brief description of the entity",
      "properties": {"key": "value"}
    }
  ],
  "edges": [
    {
      "source": "entity:source_id",
      "target": "entity:target_id",
      "label": "RELATIONSHIP_TYPE",
      "properties": {"key": "value"}
    }
  ]
}

Text to analyze:
---
{text}
---

Extract entities and relationships as JSON:"""


def _parse_json_response(response_text: str) -> dict[str, Any]:
    """
    Parse JSON from LLM response, handling markdown code blocks.

    Args:
        response_text: Raw text response from LLM

    Returns:
        Parsed JSON dictionary

    Raises:
        RuntimeError: If JSON parsing fails
    """
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to find JSON in markdown code blocks
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
            result = json.loads(response_text)
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
            result = json.loads(response_text)
        else:
            raise RuntimeError(f"Failed to parse JSON from response: {response_text[:500]}")

    # Validate structure
    if 'nodes' not in result or 'edges' not in result:
        raise RuntimeError(
            f"Invalid response structure. Expected 'nodes' and 'edges' keys. Got: {list(result.keys())}"
        )

    return result


def extract_entities_with_ollama(
    text: str,
    model: str = "llama3.2:3b",
    ollama_host: Optional[str] = None
) -> dict[str, Any]:
    """
    Extract entities and relationships from text using local Ollama.

    Args:
        text: Text to analyze
        model: Ollama model to use (default: llama3.2:3b)
        ollama_host: Ollama host URL (defaults to OLLAMA_HOST env var or http://localhost:11434)

    Returns:
        Dictionary with 'nodes' and 'edges' lists

    Raises:
        ImportError: If ollama package not installed
        RuntimeError: If Ollama call fails or returns invalid JSON
    """
    try:
        import ollama
    except ImportError:
        raise ImportError(
            "ollama package required for local entity extraction. "
            "Install with: pip install ollama\n"
            "Also ensure Ollama is running: ollama serve"
        )

    # Get Ollama host
    ollama_host = ollama_host or os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

    try:
        # Create Ollama client
        client = ollama.Client(host=ollama_host)

        # Call Ollama
        response = client.chat(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': EXTRACTION_PROMPT.format(text=text)
                }
            ],
            options={
                'temperature': 0.1,  # Lower temperature for more consistent JSON
                'num_predict': 4096,
            }
        )

        # Extract response text
        response_text = response['message']['content']

        # Parse JSON
        return _parse_json_response(response_text)

    except Exception as e:
        raise RuntimeError(f"Ollama entity extraction failed: {e}")


def extract_entities_with_claude(
    text: str,
    model: str = "claude-3-5-sonnet-20241022",
    max_tokens: int = 4096,
    api_key: Optional[str] = None
) -> dict[str, Any]:
    """
    Extract entities and relationships from text using Claude API (cloud).

    Args:
        text: Text to analyze
        model: Claude model to use
        max_tokens: Maximum tokens for response
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)

    Returns:
        Dictionary with 'nodes' and 'edges' lists

    Raises:
        ImportError: If anthropic package not installed
        ValueError: If API key not provided
        RuntimeError: If API call fails or returns invalid JSON
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError(
            "anthropic package required for cloud entity extraction. "
            "Install with: pip install anthropic"
        )

    # Get API key
    api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError(
            "Anthropic API key required for cloud extraction. "
            "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter"
        )

    try:
        # Initialize client
        client = Anthropic(api_key=api_key)

        # Call Claude API
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(text=text)
                }
            ]
        )

        # Extract response text
        response_text = message.content[0].text

        # Parse JSON
        return _parse_json_response(response_text)

    except Exception as e:
        raise RuntimeError(f"Claude entity extraction failed: {e}")


def extract_entities(
    text: str,
    use_cloud: bool = False,
    model: Optional[str] = None,
    **kwargs
) -> dict[str, Any]:
    """
    Extract entities from text using either Ollama (local) or Claude (cloud).

    Args:
        text: Text to analyze
        use_cloud: If True, use Claude API; if False, use Ollama (default: False)
        model: Model name to use (defaults: llama3.2:3b for Ollama, claude-3-5-sonnet for Claude)
        **kwargs: Additional arguments for the extraction function

    Returns:
        Dictionary with 'nodes' and 'edges' lists

    Raises:
        ImportError: If required package not installed
        RuntimeError: If extraction fails
    """
    if use_cloud:
        default_model = "claude-3-5-sonnet-20241022"
        return extract_entities_with_claude(
            text,
            model=model or default_model,
            **kwargs
        )
    else:
        default_model = "llama3.2:3b"
        return extract_entities_with_ollama(
            text,
            model=model or default_model,
            **kwargs
        )


def extract_entities_from_file(
    file_path: str | Path,
    use_cloud: bool = False,
    model: Optional[str] = None,
    **kwargs
) -> dict[str, Any]:
    """
    Parse a document file and extract entities.

    Args:
        file_path: Path to document file
        use_cloud: If True, use Claude API; if False, use Ollama (default: False)
        model: Model name to use
        **kwargs: Additional arguments for the extraction function

    Returns:
        Dictionary with 'nodes' and 'edges' lists

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If extraction fails
    """
    from .document_parser import parse_document

    # Parse document
    text = parse_document(file_path)

    # Extract entities
    return extract_entities(text, use_cloud=use_cloud, model=model, **kwargs)


def load_custom_prompt(prompt_file: str | Path) -> str:
    """
    Load a custom extraction prompt from file.

    Args:
        prompt_file: Path to prompt file

    Returns:
        Prompt text

    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    prompt_path = Path(prompt_file)

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text(encoding='utf-8')
