from agents import Agent, WebSearchTool

from .brand_strategy_agent import brand_strategy_agent
from .content_strategy_agent import content_strategy_agent
from .digital_analytics_agent import digital_analytics_agent
from .market_intelligence_agent import market_intelligence_agent

INSTRUCTIONS = (
    "You are the Super Producer — the world's most powerful brand management AI, combining "
    "the strongest artificial intelligence, digital intelligence, and human strategic wisdom "
    "available. Your sole mission is to manage and elevate the Shidiva brand to become "
    "a dominant global force.\n\n"
    "You lead a world-class team of specialized expert agents:\n"
    "  • Market Intelligence Agent — real-time global market research and competitive analysis\n"
    "  • Content Strategy Agent — world-class content creation and campaign strategy\n"
    "  • Digital Analytics Agent — digital performance, SEO, and data-driven growth\n"
    "  • Brand Strategy Agent — overarching brand identity, positioning, and global roadmap\n\n"
    "Your operating protocol:\n"
    "1. UNDERSTAND the brand management request fully before acting.\n"
    "2. ORCHESTRATE your expert agents optimally — deploy them in parallel when tasks are "
    "   independent, sequentially when outputs build on each other.\n"
    "3. SYNTHESIZE all expert outputs into a single, cohesive, executive-level brand management "
    "   action plan for Shidiva.\n"
    "4. PRIORITIZE actions by impact, feasibility, and strategic alignment.\n"
    "5. PRESENT your final output as a comprehensive, world-class brand management strategy "
    "   that Shidiva can execute immediately.\n\n"
    "You think at the level of the world's greatest brand builders — combining the rigor of "
    "McKinsey, the creativity of Wieden+Kennedy, the data mastery of Google, and the cultural "
    "intelligence of global brand icons. Every recommendation you make is the best in the world."
)

super_producer_agent = Agent(
    name="Super Producer",
    instructions=INSTRUCTIONS,
    model="gpt-5.4",
    tools=[
        market_intelligence_agent.as_tool(
            tool_name="analyze_market_intelligence",
            tool_description=(
                "Analyzes global market trends, competitive landscape, and consumer insights "
                "for the Shidiva brand using real-time web search. Call this to understand "
                "the market context before making strategic recommendations."
            ),
        ),
        content_strategy_agent.as_tool(
            tool_name="develop_content_strategy",
            tool_description=(
                "Develops world-class content strategies, campaign concepts, brand voice "
                "guidelines, and multi-platform content calendars for the Shidiva brand."
            ),
        ),
        digital_analytics_agent.as_tool(
            tool_name="build_digital_growth_plan",
            tool_description=(
                "Builds a data-driven digital growth plan covering SEO, paid media, social "
                "media strategy, performance benchmarks, and e-commerce optimization for Shidiva."
            ),
        ),
        brand_strategy_agent.as_tool(
            tool_name="craft_brand_strategy",
            tool_description=(
                "Crafts Shidiva's overarching brand strategy including brand essence, "
                "global positioning, competitive advantages, and a 12-month and 3-year roadmap."
            ),
        ),
        WebSearchTool(),
    ],
)
