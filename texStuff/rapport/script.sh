#!/bin/bash
rm extraGradMain.tex
sed -n '/\\section{.*}/,/\\end{document}/p' extra_grad.tex > temp.tex;
sed '/\\end{document}.*/d' temp.tex > extraGradMain.tex
sed -i 's/section/subsection/g' extraGradMain.tex
rm temp.tex

sed -n '/\\section{.*}/,/\\end{document}/p' rapport_edited.tex > temp.tex;
sed '/\\end{document}.*/d' temp.tex > rapport_edited_main.tex
sed -i 's/section/subsection/g' rapport_edited_main.tex
rm temp.tex
