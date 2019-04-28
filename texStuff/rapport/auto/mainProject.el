(TeX-add-style-hook
 "mainProject"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("biblatex" "backend=biber" "style=numeric" "citestyle=numeric")))
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
    "biblatex")
   (LaTeX-add-bibliographies
    "tocite"))
 :latex)

