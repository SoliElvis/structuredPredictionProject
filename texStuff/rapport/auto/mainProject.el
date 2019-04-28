(TeX-add-style-hook
 "mainProject"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("biblatex" "backend=biber" "style=numeric" "citestyle=numeric")))
   (TeX-run-style-hooks
    "premath"
    "extraGradMain"
    "rapport_edited_main"
    "exp"
    "discussion"
    "conclusion"
    "biblatex")
   (LaTeX-add-bibliographies
    "tocite"))
 :latex)

