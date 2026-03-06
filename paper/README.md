# Building the Paper

## Prerequisites

1. A LaTeX distribution (TeX Live, MiKTeX, or similar)
2. The TMLR style file (`tmlr.sty`) from the official repository:
   ```bash
   wget https://raw.githubusercontent.com/JmlrOrg/tmlr-style-file/main/tmlr.sty
   ```

## Build

```bash
cd paper/
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

This produces `main.pdf`.

## Notes

- Figures are referenced from `../results/` (the experiment output directory)
- The `\documentclass[accepted]{tmlr}` option should be changed to `[preprint]` for Zenodo upload, or removed for initial submission
- For Zenodo, the compiled PDF is the primary artifact; LaTeX source is supplementary
