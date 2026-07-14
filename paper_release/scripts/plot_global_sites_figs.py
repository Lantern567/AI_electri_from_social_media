#!/usr/bin/env python3
"""Render Figs 1-3 plus the operator appendix figure from the GLOBAL
station-based supply results (data_globalsites/) into figures_globalsites/.
"""
import warnings
from pathlib import Path

import plot_cfe_geographic_portfolio_ai as P

# The base module promotes warnings to errors to catch font issues; demote
# harmless pandas deprecation warnings so fig1/fig3 render.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

REPORT = P.REPORT
P.DATA = REPORT / "data_globalsites"
P.FIG = REPORT / "figures_globalsites"
P.FIG.mkdir(parents=True, exist_ok=True)
P.set_style()

if __name__ == "__main__":
    print(f"DATA={P.DATA}\nFIG={P.FIG}")
    P.plot_fig1()
    print("fig1 ok")
    P.plot_fig2()
    print("fig2 ok")
    P.plot_fig3()
    print("fig3 ok")
    import plot_fig3_redesign as r3
    r3.render_appendix(P.DATA, P.FIG)
    print("appendix ok")
