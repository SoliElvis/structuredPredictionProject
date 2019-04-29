(TeX-add-style-hook
 "FW_review"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("icml2016" "accepted")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "times"
    "graphicx"
    "subfigure"
    "natbib"
    "algorithm"
    "algorithmic"
    "amsmath"
    "amsfonts"
    "hyperref"
    "icml2016"
    "csvsimple")
   (TeX-add-symbols
    "theHalgorithm")
   (LaTeX-add-labels
    "alg:example"))
 :latex)

