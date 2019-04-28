(TeX-add-style-hook
 "mainProject"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("biblatex" "backend=biber" "style=numeric" "citestyle=numeric") ("eucal" "mathcal")))
   (TeX-run-style-hooks
    "premath"
    "structSVM"
    "prelim_large_margin"
    "cvxanalysis"
    "extraGradMain"
    "bcfwMain"
    "exp"
    "discussion"
    "conclusion"
    "appendix"
    "biblatex"
    "eucal")
   (LaTeX-add-bibliographies
    "tocite"))
 :latex)

