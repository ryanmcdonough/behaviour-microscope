# Authority paper arXiv source

This folder is a self-contained PDFLaTeX source tree for the plain-language
version of the authority paper.

## Build locally

Run these commands from this folder:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The resulting file is `main.pdf`.

## Prepare an arXiv upload

Upload only:

- `main.tex`
- `references.bib`
- the four PNG files used by `main.tex` under `figures/`

Select PDFLaTeX if arXiv does not detect it automatically. Do not upload local
build products such as `main.pdf`, `main.aux`, `main.log`, `main.out`,
`main.bbl`, or `main.toc` unless arXiv reports a bibliography-processing problem.
The document uses a fixed version date so a later arXiv rebuild will not change
the date printed in the paper.

Before submission, replace the availability text with a permanent repository or
archive link tied to the submitted commit and recheck the time-sensitive legal
scenario.
