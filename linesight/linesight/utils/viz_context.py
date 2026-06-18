"""
viz_context.py
--------------
Prints a structured pedagogical context block after every LineSight visualization.
Called from every plot_*.py / animate_*.py file just before returning the figure.
"""

_SEP  = "─" * 62
_SEP2 = "═" * 62


def print_viz_context(
    *,
    diagram: str,
    solves: str,
    theory: str,
    formula: str,
    constraints: str,
    reading: str,
):
    """
    Print a uniform 'Context Panel' under every diagram.

    Parameters
    ----------
    diagram      : Short title of what the diagram IS.
    solves       : One-sentence answer to "what problem does this solve?"
    theory       : Plain-English description of the underlying math concept.
    formula      : The key mathematical formula, written as a plain string.
    constraints  : Requirements / assumptions the technique relies on.
    reading      : How to READ / interpret the chart in one or two lines.
    """
    print()
    print(_SEP2)
    print(f"  DIAGRAM  : {diagram}")
    print(_SEP)
    print(f"  SOLVES   : {solves}")
    print(_SEP)
    print(f"  THEORY   : {theory}")
    print(_SEP)
    print(f"  FORMULA  : {formula}")
    print(_SEP)
    print(f"  CONSTRAINTS :")
    for line in constraints.strip().splitlines():
        print(f"    {line.strip()}")
    print(_SEP)
    print(f"  HOW TO READ :")
    for line in reading.strip().splitlines():
        print(f"    {line.strip()}")
    print(_SEP2)
    print()
