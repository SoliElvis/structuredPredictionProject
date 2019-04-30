(TeX-add-style-hook
 "mainProject"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("biblatex" "backend=biber" "citestyle=authoryear-ibid" "natbib=true") ("appendix" "toc" "page")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "premath"
    "intro"
    "structSVM"
    "extraGradMain"
    "bcfwMain"
    "exp"
    "cvxanalysis"
    "app_extra"
    "app_fw"
    "graphicx"
    "biblatex"
    "csquotes"
    "comment"
    "fancyhdr"
    "appendix")
   (LaTeX-add-bibliographies
    "tocite.bib"
    "tocite"))
 :latex)

