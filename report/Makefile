TEX_MAIN = neural3.tex

all: neural3

neural3: $(TEX_MAIN)
	latexmk -pdf  $(TEX_MAIN) 

clean:
	latexmk -C $(TEX_MAIN)
	rm -f *.bbl 