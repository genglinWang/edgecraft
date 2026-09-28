#!/usr/bin/env python3
"""Main entry point for EdgeCraft CLI.

This module provides the main entry point for the EdgeCraft command-line interface.
It imports and runs the main CLI function from the commands module.
"""

if __name__ == "__main__":
    from edgecraft.cli.commands import main
    main()