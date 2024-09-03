# Define default rule
all:
#	pdflatex cover_letter.tex
#	pdflatex cover_letter.tex
	pdflatex cover_letter_2.tex
	pdflatex cover_letter_2.tex
#	pdflatex main.tex
#	pdflatex main.tex
#	biber main
#	pdflatex main.tex
#	pdflatex main.tex
	pdflatex main_2.tex
	pdflatex main_2.tex
	biber main_2
	pdflatex main_2.tex
	pdflatex main_2.tex
	latexdiff main.tex main_2.tex > diff.tex
	pdflatex diff.tex
	pdflatex diff.tex
	biber diff
	pdflatex diff.tex
	pdflatex diff.tex

# Define clean rule
clean:
	rm -rf *.aux *.bbl *.bcf *.blg *.log *.out *.run.xml *fff *gz *ttt

.PHONY: all clean
