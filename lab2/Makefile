CC = mpicc
CFLAGS =
LIBS =
SOURCE = grid
TARGETS = grid_4_4 grid_512_512

default: clean $(TARGETS)
all: default

.PHONY: clean

grid_4_4:
	$(CC) $(SOURCE).c -o grid_4_4 $(CFLAGS) -std=c99 $(LIBS) -D RUNTIME_SMALL

grid_512_512:
	$(CC) $(SOURCE).c -o grid_512_512 $(CFLAGS) -std=c99 $(LIBS)

clean:
	-rm -f $(TARGETS)
	-rm -f *.o
