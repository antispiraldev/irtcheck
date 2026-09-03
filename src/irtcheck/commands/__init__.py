"""One module per command, each owned by exactly one wave-1 brief.

cli.py declares the flags; these modules do the work. The split is what lets
four agents build wave 1 concurrently without sharing a file — see cli.py's
module docstring.
"""
