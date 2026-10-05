"""Module entrypoint for nanogent."""

import asyncio

from nanogent.cli import main


if __name__ == "__main__":
    asyncio.run(main())
