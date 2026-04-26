from pydantic import BaseModel

from agents import Agent, ModelSettings, WebSearchTool

INSTRUCTIONS = (
    "You are a globally recognized brand strategist who has built some of the world's most "
    "valuable brands. Your role is to develop the overarching brand strategy for Shidiva, "
    "positioning it as a world-class brand with a compelling identity, clear differentiation, "
    "and a long-term vision for global market leadership. "
    "\n\n"
    "When crafting brand strategy, you:\n"
    "1. Define Shidiva's brand essence: purpose, vision, mission, and core values that "
    "   differentiate it powerfully in the global market.\n"
    "2. Develop the positioning statement and value proposition for primary and secondary "
    "   target audiences across key global markets.\n"
    "3. Map the competitive landscape and articulate Shidiva's unique advantages and "
    "   defensible moats against global competitors.\n"
    "4. Design a brand architecture framework for product lines, extensions, and partnerships.\n"
    "5. Create a 12-month and 3-year brand roadmap with milestones for brand equity growth, "
    "   market expansion, and revenue targets.\n"
    "6. Identify strategic partnership, licensing, and collaboration opportunities with "
    "   global powerhouses that elevate Shidiva's stature.\n"
    "\n"
    "Your output is strategic, visionary, and grounded in global brand-building excellence."
)


class BrandStrategyReport(BaseModel):
    brand_essence: str
    """Shidiva's core purpose, vision, mission, and values."""

    positioning_statement: str
    """Concise global positioning statement for Shidiva."""

    competitive_advantages: list[str]
    """Key differentiators that set Shidiva apart from global competitors."""

    roadmap_12_months: list[str]
    """Priority initiatives for the next 12 months."""

    roadmap_3_years: list[str]
    """Strategic milestones for the 3-year brand growth plan."""

    partnership_opportunities: list[str]
    """High-impact partnership and collaboration recommendations."""

    executive_summary: str
    """A powerful 3-5 sentence executive summary of the full brand strategy."""


brand_strategy_agent = Agent(
    name="Brand Strategy Agent",
    handoff_description=(
        "World-class brand strategist that develops Shidiva's overarching brand identity, "
        "positioning, competitive advantages, and long-term global growth roadmap."
    ),
    instructions=INSTRUCTIONS,
    model="gpt-5.4",
    model_settings=ModelSettings(temperature=0.7),
    output_type=BrandStrategyReport,
    tools=[WebSearchTool()],
)
