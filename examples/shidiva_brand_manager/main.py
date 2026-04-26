import asyncio

from examples.auto_mode import input_with_fallback

from .manager import ShidivaBrandManager

DEFAULT_REQUEST = (
    "Conduct a full brand management review for Shidiva. "
    "Analyze the global market, build a world-class content strategy, "
    "develop a digital growth plan, and craft a comprehensive brand strategy "
    "with a 12-month action roadmap."
)


async def main() -> None:
    request = input_with_fallback(
        "What would you like the Super Producer to do for the Shidiva brand? ",
        DEFAULT_REQUEST,
    )
    await ShidivaBrandManager().run(request)


if __name__ == "__main__":
    asyncio.run(main())
