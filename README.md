# Chasing wind and sun narrows but does not close the data-centre clean-power gap

Official code and data repository for the manuscript **“Chasing wind and sun narrows but does not close the data-centre clean-power gap.”**

The submission-ready package is in [`paper_release/`](paper_release/). It contains source and derived data, analysis scripts, frozen figures, and a parameter–script–input–output crosswalk.

## Reproduce the paper

1. Create the Python 3.12 environment with `uv sync`.
2. Follow [`paper_release/README.md`](paper_release/README.md).
3. Trace results with [`paper_release/REPRODUCIBILITY_INDEX.md`](paper_release/REPRODUCIBILITY_INDEX.md).

## Repository scope

This public repository is intentionally limited to the manuscript and its reproducibility package. The broader development workspace, unrelated projects, caches and exploratory outputs are not included. Large third-party raw datasets that cannot be redistributed remain available from the original providers listed in the manuscript; redistributable figure source data and derived results are included in `paper_release/`.

## Submission version

The submitted code-and-data snapshot is tagged `v1.1.0-submission`. Use that tag, rather than the moving `main` branch, when reproducing the submitted manuscript. A Zenodo DOI can be added after linking this repository to Zenodo and archiving the tagged version.

## Licence

Source code is released under the MIT License; see [`LICENSE`](LICENSE). Newly generated derived data are released under CC BY 4.0 subject to the qualifications in [`DATA_LICENSE.md`](DATA_LICENSE.md). Third-party inputs retain their original licences and terms.
