"""Headless smoke test for the Universal Credit Studio UI."""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "auto_app.py"


def main() -> int:
    app = AppTest.from_file(str(APP), default_timeout=300)
    app.run()

    if app.exception:
        print("EXCEPTION during initial render:")
        for item in app.exception:
            print(item.value)
        return 1

    print(f"initial render OK - tabs={len(app.tabs)} buttons={len(app.button)} "
          f"selectboxes={len(app.selectbox)}")

    run_button = next((b for b in app.button if "Run auto" in str(b.label)), None)
    if run_button is None:
        print("could not find the run button:", [b.label for b in app.button])
        return 1

    run_button.click()
    app.run()

    if app.exception:
        print("EXCEPTION during pipeline run:")
        for item in app.exception:
            print(item.value)
        return 1

    print("pipeline run OK")
    print(f"  metrics: {len(app.metric)} dataframes: {len(app.dataframe)} "
          f"plotly: {len(app.get('plotly_chart'))}")

    successes = [s.value for s in app.success]
    print(f"  success boxes: {len(successes)}")
    for message in successes[:2]:
        print("   -", message[:160].replace("\n", " "))

    errors = [e.value for e in app.error]
    if errors:
        print("  ERRORS:", errors)
        return 1

    tabs = [t.label for t in app.tabs]
    print(f"  tabs: {tabs}")

    if not any("Selected" in s for s in successes):
        print("  model selection banner missing")
        return 1

    # ---- upload tab: format table + example download ---------------------
    data_radio = next((r for r in app.radio if r.label == "Data"), None)
    if data_radio is None:
        print("  data source radio not found:", [r.label for r in app.radio])
        return 1

    data_radio.set_value("Upload CSVs").run()
    if app.exception:
        print("EXCEPTION on the upload tab:")
        for item in app.exception:
            print(item.value)
        return 1

    downloads = [d.label for d in app.get("download_button")]
    print(f"upload tab OK - dataframes={len(app.dataframe)} downloads={len(downloads)}")
    print("  downloads:", downloads[:8])

    if not any("example dataset" in label.lower() for label in downloads):
        print("  example dataset download missing")
        return 1
    if not any("template" in label.lower() for label in downloads):
        print("  blank template download missing")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
