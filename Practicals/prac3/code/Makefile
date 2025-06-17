

all: barley barley_ring barley_combi

barley:		barley.c Makefile
	icc barley.c -o barley $(INC) $(LIB)

barley_ring:	barley_ring.c Makefile
	icc barley_ring.c -o barley_ring $(INC) $(LIB)

barley_combi: 	barley_combi.c Makefile
	icc barley_combi.c -o barley_combi $(INC) $(LIB)

clean:
	rm -f barley barley_ring barley_combi

