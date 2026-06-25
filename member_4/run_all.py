"""Single entry point: regenerate Figures 9-14 of Fujie & Odagaki (2007) end
to end.

    python3 run_all.py

Produces, under ./data/ (raw CSVs) and ./figures/ (PNGs):
    fig09_route_strong_infectiousness.png
    fig10_route_hub_model.png
    fig11_route_no_superspreaders.png
    fig12_link_distribution_no_superspreaders.png
    fig13_link_distribution_with_superspreaders.png
    fig14_sars_comparison.png
"""

import make_figure_14
import make_figures_9_11
import make_figures_12_13


def main():
    print("=" * 70)
    print("Figures 9-11: spatial route-of-infection maps")
    print("=" * 70)
    make_figures_9_11.main()

    print()
    print("=" * 70)
    print("Figures 12-13: secondary-infection ('number of links') distributions")
    print("=" * 70)
    prob0, prob_strong, prob_hub = make_figures_12_13.main()

    print()
    print("=" * 70)
    print("Figure 14: SARS Singapore data vs. model distributions")
    print("=" * 70)
    make_figure_14.main(prob_strong=prob_strong, prob_hub=prob_hub)

    print()
    print(
        "Done. See ./figures/ for the PNGs and ./data/ for the saved "
        "coordinates / infection logs / secondary-infection counts."
    )


if __name__ == "__main__":
    main()
