import numpy as np
import pandas as pd
import subprocess
from pathlib import Path

from Esercizio_CLOUDY_griglia import (
    ROOT,
    CLOUDY_EXE,
    CFG,
    LOG_RADIUS,
    LOG_N_H,
    LOG_COLUMN,
    LINES,
    build_unified_spectrum,
    save_sed,
    parse_intensita_cloudy,
    extract_line,
)


# ============================================================
# TEST COVERING FACTOR: Cf = 0.1 vs Cf = 0.2
# Output completamente separati dalla griglia originale
# ============================================================

TEST_DIR = ROOT / "test_covering_factor"
TEST_DIR.mkdir(parents=True, exist_ok=True)

# Una sola combinazione di massa e mdot
TEST_MASS = 1e8
TEST_MDOT = 0.1
TEST_CFS = [0.1, 0.2]

# Genera una sola volta la SED
nu, lnu, _, info = build_unified_spectrum(
    np.log10(TEST_MASS),
    np.log10(TEST_MDOT),
    CFG
)

regime = info["regime"]

results = []

for cf in TEST_CFS:

    # Cartella indipendente per ciascun valore di Cf
    folder = TEST_DIR / f"cf_{cf:.1f}"
    folder.mkdir(parents=True, exist_ok=True)

    stem = f"test_cf_{cf:.1f}"

    # Salva la SED
    lint = save_sed(nu, lnu, folder / f"{stem}.sed")

    # Input CLOUDY
    commands = [
        f"title Test covering factor Cf={cf}",
        f'table SED "{stem}.sed"',
        f"luminosity total {np.log10(lint):.9f}",
        f"radius {LOG_RADIUS:.4f}",
        f"hden {LOG_N_H:.4f}",
        f"stop column density {LOG_COLUMN:.4f}",
        f"covering factor {cf:.6f}",
        "iterate",
        f'save last lines intensity "{stem}.int" emergent column all',
        f'save last lines labels "{stem}.labels"',
        "database print",
        "print citation",
    ]

    (folder / f"{stem}.in").write_text(
        "\n".join(commands) + "\n"
    )

    # Esegui CLOUDY
    run = subprocess.run(
        [str(CLOUDY_EXE), "-r", stem],
        cwd=folder,
        capture_output=True,
        text=True
    )

    out_path = folder / f"{stem}.out"
    out_text = (
        out_path.read_text(errors="replace")
        if out_path.exists()
        else ""
    )

    if run.returncode != 0 or "Cloudy exited OK" not in out_text:
        raise RuntimeError(
            f"CLOUDY fallito per Cf={cf}\n"
            + run.stderr[-2000:]
            + "\n"
            + "\n".join(out_text.splitlines()[-30:])
        )

    # Leggi le intensità RAW del file .int
    df = parse_intensita_cloudy(
        folder / f"{stem}.int"
    )

    for name, spec in LINES.items():

        intensity = extract_line(df, name, spec)

        results.append({
            "Riga": name,
            "Cf": cf,
            "I_area": intensity
        })

# ============================================================
# CONFRONTO DELLE INTENSITÀ RAW
# ============================================================

results_df = pd.DataFrame(results)

comparison = results_df.pivot(
    index="Riga",
    columns="Cf",
    values="I_area"
)

comparison.columns = [
    f"I_Cf_{cf:.1f}" for cf in comparison.columns
]

comparison["Rapporto_0.2_0.1"] = (
    comparison["I_Cf_0.2"] /
    comparison["I_Cf_0.1"].replace(0, np.nan)
)

print("\nCONFRONTO INTENSITÀ CLOUDY NON SCALATE")
print(f"M = {TEST_MASS:.2e} Msun")
print(f"mdot = {TEST_MDOT}")
print(f"Regime = {regime}")
print()

print(comparison.to_string())

# Salva il confronto separatamente
comparison.to_csv(
    TEST_DIR / "confronto_covering_factor.csv"
)

print("\nRisultati salvati in:", TEST_DIR)
