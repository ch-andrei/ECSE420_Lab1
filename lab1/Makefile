CC = mpicc
CFLAGS =
LIBS =
TARGETS = rectifyMPI

default: clean $(TARGETS)
all: default

.PHONY: clean

rectifyMPI:
	$(CC) -o rectify rectifyMPI.c lodepng.c $(CFLAGS) -std=c99 $(LIBS)

clean:
	-rm -f $(TARGETS)
	-rm -f *.o
