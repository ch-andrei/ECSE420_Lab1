TARGET = rectify
CC = gcc
CFLAGS = -g -Wall
LIBS = -lpthread

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = rectify.c lodepng.c
HEADERS = lodepng.h wm.h

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -std=c99 -Wall $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)