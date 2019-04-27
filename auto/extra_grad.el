(TeX-add-style-hook
 "extra_grad"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("geometry" "margin=1in")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "amsmath"
    "amsfonts"
    "amssymb"
    "geometry"
    "algorithmic")
   (TeX-add-symbols
    "argmin"
    "argmax")
   (LaTeX-add-labels
    "eq1"
    "saddle_point"
    "saddle_obj"))
 :latex)

