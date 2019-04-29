(TeX-add-style-hook
 "mainProject"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("biblatex" "backend=biber" "citestyle=authoryear-ibid" "natbib=true")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (TeX-run-style-hooks
    "premath"
    "intro"
    "structSVM"
    "cvxanalysis"
    "extraGradMain"
    "bcfwMain"
    "exp"
    "appendix"
    "biblatex"
    "csquotes"
    "fancyhdr")
   (LaTeX-add-bibliographies
    "tocite.bib"
    "tocite"))
 :latex)

