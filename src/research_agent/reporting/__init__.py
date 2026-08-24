"""
reporting/ -- presentation-layer report generators built on top of
log_event()'s stream of events, never a second instrumentation path.

What lives here today:
    narrative.py   the human-readable execution-narrative renderer
                   (S-2), split out of logging_setup.py because it was
                   an import-order coupling on the project's most
                   fundamental utility for a feature most call sites
                   never use.
"""
