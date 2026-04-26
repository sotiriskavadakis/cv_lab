.PHONY: build clean

build:
	lualatex -interaction=nonstopmode -file-line-error -shell-escape report.tex
	lualatex -interaction=nonstopmode -file-line-error -shell-escape report.tex

clean:
	rm -f *.aux *.log *.out *.toc *.pdf
