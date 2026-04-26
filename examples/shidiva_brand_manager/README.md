# Shidiva Brand Manager — Super Producer

A world-class, multi-agent brand management system for the **Shidiva** brand, powered by the
strongest combination of **Artificial Intelligence**, **Digital Intelligence**, and
**Human Strategic Wisdom** available on the global market.

## Overview

This example demonstrates an advanced orchestrated multi-agent system built with the
OpenAI Agents Python SDK. A **Super Producer** orchestrator coordinates a team of four
elite specialist agents to deliver comprehensive brand management at a world-class level.

```
Super Producer (Orchestrator)
├── Market Intelligence Agent   ← Global trends & competitive analysis (web search)
├── Content Strategy Agent       ← Campaigns, brand voice, content calendars (web search)
├── Digital Analytics Agent      ← SEO, paid media, digital growth (web search)
└── Brand Strategy Agent         ← Brand essence, positioning, roadmap (structured output)
```

## Agent Capabilities

| Agent | Intelligence | Tools |
|---|---|---|
| **Super Producer** | Orchestration + synthesis | All specialist tools + web search |
| **Market Intelligence** | Global market analysis | WebSearchTool |
| **Content Strategy** | Campaign & content creation | WebSearchTool |
| **Digital Analytics** | Data-driven digital growth | WebSearchTool |
| **Brand Strategy** | Brand identity & roadmap | WebSearchTool + structured output |

## Patterns Demonstrated

- **Agents as Tools** — specialist agents called as tools by the Super Producer
- **Parallel Orchestration** — multiple specialist agents can run concurrently
- **Structured Outputs** — Brand Strategy Agent returns a typed `BrandStrategyReport`
- **Streaming** — results stream progressively with a live progress indicator
- **Tracing** — full observability via OpenAI Traces

## Usage

```bash
python -m examples.shidiva_brand_manager.main
```

Or run with a custom brand management request:

```python
from examples.shidiva_brand_manager.manager import ShidivaBrandManager
import asyncio

asyncio.run(ShidivaBrandManager().run(
    "Build a Q4 global launch campaign strategy for Shidiva's new product line."
))
```

## Example Requests

- `"Analyze Shidiva's competitive position and build a 12-month brand growth roadmap."`
- `"Develop a global content strategy for Shidiva targeting Gen Z across TikTok and Instagram."`
- `"Build a digital performance plan for Shidiva to double online revenue in 12 months."`
- `"Craft Shidiva's brand identity from scratch for global market entry."`
