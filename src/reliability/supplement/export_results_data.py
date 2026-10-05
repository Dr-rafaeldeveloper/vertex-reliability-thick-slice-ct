"""exports as CSV the per-case data that support the plots of
Section 3, from the results JSONs (current hash). Outside the package hash (subfolder).
Output: output/figures/results_figure_data/*.csv + README.md + _provenance.json.
Usage: python src/reliability/supplement/export_results_data.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from reliability import a4_config as C  # noqa: E402

DOC = os.path.join(C.ROOT, "output", "figures")
OUT = os.path.join(DOC, "results_figure_data")


def J(name):
    p = os.path.join(C.A4_RESULTS, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def write(name, header, rows, desc, source, listing):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    listing.append((name, desc, source, len(rows)))


def main():
    os.makedirs(OUT, exist_ok=True)
    done = []
    r31, r32, r33 = (
        J("r31_reconstruction_error.json"),
        J("r32_association.json"),
        J("r33_localization_calibration.json"),
    )
    ids = sorted(r31["per_case"])
    m33 = (
        "rho_vertex",
        "rho_region",
        "auroc_decile",
        "precision_10",
        "flagged_error_mm",
        "residual_error_mm",
        "mae_mm",
        "constant_mae_mm",
        "slope",
        "intercept_mm",
    )
    write(
        "foot_per_case.csv",
        [
            "foot",
            "e_median_sr_mm",
            "e_median_tri_mm",
            "rho_vertex_field_sr",
            "rho_vertex_field_tri",
        ]
        + [m for m in m33 if m != "rho_vertex"],
        [
            [
                h,
                r31["per_case"][h]["sr"]["median_mm"],
                r31["per_case"][h]["trilinear"]["median_mm"],
                r32["per_case_sr"][h]["rho_vertex"],
                r32["per_case_trilinear"][h]["rho_vertex"],
            ]
            + [r33["per_case_sr"][h][m] for m in m33 if m != "rho_vertex"]
            for h in ids
        ],
        "Metrics per foot (48): SR and trilinear reconstruction error, field correlation, localization and calibration",
        "r31, r32, r33",
        done,
    )
    write(
        "foot_calibration_deciles.csv",
        [
            "decile",
            "predicted_median_mm",
            "predicted_q1",
            "predicted_q3",
            "observed_median_mm",
            "observed_q1",
            "observed_q3",
            "n_feet",
        ],
        [
            [
                d["decile"],
                d["predicted_mm"]["median"],
                *d["predicted_mm"]["iqr"],
                d["observed_mm"]["median"],
                *d["observed_mm"]["iqr"],
                d["predicted_mm"]["n"],
            ]
            for d in r33["sr_deciles"]
        ],
        "Calibration by deciles of the predicted error (median and IQR across feet)",
        "r33.sr_deciles",
        done,
    )
    rows = []
    for fname in ("r341_intensity_uncertainty.json", "r342_surface_uncertainty.json"):
        j = J(fname)
        for method, pc in j["per_case"].items():
            for h, v in pc.items():
                rows.append(
                    [
                        method,
                        h,
                        v.get("rho_vertex"),
                        v.get("rho_region"),
                        v.get("auroc_decile"),
                    ]
                )
    for h in ids:
        rows.append(
            [
                "field",
                h,
                r32["per_case_sr"][h]["rho_vertex"],
                r33["per_case_sr"][h]["rho_region"],
                r33["per_case_sr"][h]["auroc_decile"],
            ]
        )
    write(
        "foot_uncertainties_vs_field.csv",
        ["method", "foot", "rho_vertex", "rho_region", "auroc_decile"],
        rows,
        "Correlation and AUROC per foot of each uncertainty (u_ens, u_mc, u_dropens, u_ds, u_geo_*) and of the field",
        "r341, r342, r32, r33",
        done,
    )
    r35 = J("r35_ablation.json")
    rows = []
    for group in ("isolated", "without_one"):
        for f_, v in r35[group].items():
            res = v["summary"] if "summary" in v else v
            rv = res["rho_vertex"]
            extra = []
            if group == "without_one":
                d = v["delta_rho"]
                extra = [
                    d["median"],
                    d["iqr"][0],
                    d["iqr"][1],
                    v["comparison_full_vs_without"]["p_bh"],
                ]
            else:
                extra = ["", "", "", ""]
            rows.append(
                [group, f_, rv["median"], rv["iqr"][0], rv["iqr"][1]] + extra
            )
    write(
        "foot_ablation.csv",
        [
            "model",
            "feature",
            "rho_vertex_median",
            "rho_q1",
            "rho_q3",
            "delta_rho_median",
            "delta_q1",
            "delta_q3",
            "p_bh",
        ],
        rows,
        "Ablation: each feature alone and the model without it (Δρ = full − reduced; BH m = 9)",
        "r35",
        done,
    )
    r37 = J("r37_registration.json")
    ests = list(r37["strategies"])
    write(
        "foot_registration_per_case.csv",
        ["foot"] + ["%s_median_mm" % s for s in ests],
        [[h] + [r37["per_case"][h][s]["median_mm"] for s in ests] for h in ids],
        "Guided registration: error at the remote target (median of 300 trials) per foot and strategy",
        "r37.per_case",
        done,
    )
    r36 = J("r36_thorax.json")
    if r36 is not None:  # the thoracic analysis may still be pending when the foot CSVs are exported
        tids = sorted(r36["per_case_sr"])
        write(
            "thorax_per_case.csv",
            ["case"] + list(m33),
            [[h] + [r36["per_case_sr"][h][m] for m in m33] for h in tids],
            "Thorax (100 test cases): field metrics per case",
            "r36.per_case_sr",
            done,
        )
    rc = J("rC_sensitivity.json")
    if rc is not None:
        rows = []
        for name, r in rc["summary"].items():
            for m in (
                "e_median_sr",
                "e_median_tri",
                "rho_vertex",
                "rho_region",
                "auroc_decile",
            ):
                d = r[m].get("diff_vs_base", {})
                rows.append(
                    [
                        name,
                        m,
                        r[m]["median"],
                        *r[m]["iqr"],
                        d.get("median", ""),
                        d.get("p_wilcoxon", ""),
                    ]
                )
        write(
            "sensitivity.csv",
            [
                "variant",
                "metric",
                "median",
                "q1",
                "q3",
                "diff_vs_base_median",
                "p_wilcoxon",
            ],
            rows,
            "Sensitivity to the choices without a standard in the literature (foot)",
            "rC_sensitivity",
            done,
        )
    json.dump(
        {
            "generator": "src/reliability/supplement/export_results_data.py",
            "environment": C.environment_record(with_torch=False),
        },
        open(os.path.join(OUT, "_provenance.json"), "w", encoding="utf-8"),
        indent=1,
        ensure_ascii=False,
    )
    read = [
        "# Data for the results figures",
        "",
        "Generated by `src/reliability/supplement/export_results_data.py` (package hash in `_provenance.json`). "
        "Each CSV holds the per-case values behind one plot of Section 3. "
        "Data of the method figures: `figs/data/`. Selections: `supplement_rf_selection/`, "
        "`supplement_sr_selection/`.",
        "",
        "| file | content | source | rows |",
        "|---|---|---|---|",
    ] + ["| `%s` | %s | %s | %d |" % f for f in done]
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(
        "\n".join(read) + "\n"
    )
    print("exported:", [f[0] for f in done], "->", OUT)


if __name__ == "__main__":
    main()
