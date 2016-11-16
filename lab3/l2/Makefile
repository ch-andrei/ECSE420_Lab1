CC = nvcc
CFLAGS =
LIBS =
SOURCE = grid
TARGETS = grid_512_512

default: clean $(TARGETS)
all: default

.PHONY: clean

grid_4_4:
	$(CC) $(SOURCE).cu -o grid_4_4 $(CFLAGS) $(LIBS) -D RUNTIME_SMALL

grid_512_512:
	$(CC) $(SOURCE).cu -o grid_512_512 $(CFLAGS) $(LIBS)

debug_grid:
	$(CC) $(SOURCE).cu -o debug_grid $(CFLAGS) $(LIBS) -D DEBUG

clean:
	-rm -f $(TARGETS)
	-rm -f *.o
