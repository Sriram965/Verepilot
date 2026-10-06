from __future__ import annotations


LLM_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": (
                "Navigate the browser to an allowed "
                "benchmark URL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "Destination URL."
                        ),
                    },
                },
                "required": [
                    "url",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_state",
            "description": (
                "Capture the current browser state."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": (
                "Click an interactive element from "
                "the latest browser observation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "element_id": {
                        "type": "string",
                        "description": (
                            "Observation-scoped "
                            "element ID."
                        ),
                    },
                    "observation_id": {
                        "type": "string",
                        "description": (
                            "ID of the observation "
                            "used to choose the element."
                        ),
                    },
                },
                "required": [
                    "element_id",
                    "observation_id",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type",
            "description": (
                "Replace the contents of a supported "
                "text input with the supplied text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "element_id": {
                        "type": "string",
                        "description": (
                            "Observation-scoped "
                            "element ID."
                        ),
                    },
                    "text": {
                        "type": "string",
                        "description": (
                            "Text to enter."
                        ),
                    },
                    "observation_id": {
                        "type": "string",
                        "description": (
                            "ID of the observation "
                            "used to choose the element."
                        ),
                    },
                },
                "required": [
                    "element_id",
                    "text",
                    "observation_id",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press",
            "description": (
                "Press an allowed keyboard key or "
                "key combination."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": (
                            "Keyboard key or combination."
                        ),
                    },
                    "observation_id": {
                        "type": "string",
                        "description": (
                            "ID of the observation "
                            "from which this action "
                            "was selected."
                        ),
                    },
                },
                "required": [
                    "key",
                    "observation_id",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "go_back",
            "description": (
                "Navigate the browser back to the "
                "previous page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "observation_id": {
                        "type": "string",
                        "description": (
                            "ID of the observation "
                            "from which this action "
                            "was selected."
                        ),
                    },
                },
                "required": [
                    "observation_id",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Declare that you believe the task "
                "has been completed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": (
                            "Concise statement of "
                            "why you believe the task "
                            "is complete."
                        ),
                    },
                },
                "required": [
                    "claim",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "give_up",
            "description": (
                "Declare that you cannot complete "
                "the task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Reason the task cannot "
                            "be completed."
                        ),
                    },
                },
                "required": [
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
    },
]