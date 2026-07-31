"""Entry point for `python3 -m tatara`.

Without this, `python3 -m tatara.cli` runs the module and exits silently
because `main` is never called — a trap recorded in SESSION_027.
"""

from tatara.cli import main

if __name__ == "__main__":
    main()
