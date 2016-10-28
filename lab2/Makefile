CC = mpicc
CFLAGS =
LIBS =
TARGETS = grid_4_4

default: clean $(TARGETS)
all: default

.PHONY: clean

grid_4_4:
	$(CC) -o grid $(TARGETS).c  $(CFLAGS) -std=c99 $(LIBS)

clean:
	-rm -f $(TARGETS)
	-rm -f *.o
