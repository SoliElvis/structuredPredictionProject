(TeX-add-style-hook
 "premath"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("algorithm" "plain") ("isodate" "english") ("geometry" "margin=1in") ("inputenc" "utf8") ("fontenc" "T1") ("hyperref" "bookmarks=true")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "algorithmic"
    "algorithm"
    "comment"
    "isodate"
    "graphicx"
    "geometry"
    "siunitx"
    "paracol"
    "enumitem"
    "bbm"
    "inputenc"
    "fontenc"
    "hyperref"
    "bookmark"
    "pdfpages"
    "amsmath"
    "amsfonts"
    "amssymb"
    "amsthm"
    "mathtools"
    "xparse")
   (TeX-add-symbols
    '("given" ["argument"] 0)
    '("ft" 1)
    "argmin"
    "argmax"
    "E"
    "Var"
    "Cov"
    "xiii"
    "yiii"
    "dd"
    "tran"
    "trace")
   (LaTeX-add-lengths
    "tindent")
   (LaTeX-add-amsthm-newtheorems
    "theorem"
    "lemma"
    "definition"
    "proposition")
   (LaTeX-add-mathtools-DeclarePairedDelimiters
    '("abs" "")
    '("norm" "")))
 :latex)

