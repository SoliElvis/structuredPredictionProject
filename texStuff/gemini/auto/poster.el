(TeX-add-style-hook
 "poster"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("beamer" "final")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("fontenc" "T1") ("beamerposter" "size=custom" "width=120" "height=72" "scale=1.0")))
   (add-to-list 'LaTeX-verbatim-environments-local "semiverbatim")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "beamer"
    "beamer10"
    "fontenc"
    "lmodern"
    "beamerposter"
    "graphicx"
    "booktabs"
    "tikz"
    "pgfplots")
   (TeX-add-symbols
    "argmax"
    "separatorcolumn")
   (LaTeX-add-labels
    "extragrad"
    "feat_vec")
   (LaTeX-add-bibliographies)
   (LaTeX-add-lengths
    "sepwidth"
    "colwidth"))
 :latex)

