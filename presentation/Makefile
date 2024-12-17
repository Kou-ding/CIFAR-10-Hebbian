TEX_MAIN = present.tex

all: present

present: $(TEX_MAIN)
	latexmk -pdf  $(TEX_MAIN) 

clean:
	latexmk -C $(TEX_MAIN)
	rm -f *.bbl *.nav *.snm