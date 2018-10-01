# directory path:
PWD     =  $(shell pwd)
LIBS    = -lm
CC  =  gcc
CFLAGS  = -O2
EXES    = harmosc_euler

all: $(EXES)

harmosc_euler: harmosc_euler.c
	$(CC) -o harmosc_euler $(CFLAGS) $(LDFLAGS) harmosc_euler.c $(LIBS)

clean:
	rm -f *.o *.a $(EXES);
