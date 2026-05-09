# i.hyper.atcorr — Makefile
#
# Two build modes, selected by the DEBIAN_BUILD variable:
#
#   make                    — standard GRASS Module.make build (default)
#   make DEBIAN_BUILD=1     — standalone build using libras3d + libsixsv
#
# The Debian mode produces a self-contained binary that runs without GRASS.
# It requires the following packages to be installed:
#   libras3d-dev   (apt install or dpkg -i from ras3d/debian)
#   libsixsv-dev   (apt install or dpkg -i from libsixsv/debian)

# ── Debian standalone build ────────────────────────────────────────────────────
ifeq ($(DEBIAN_BUILD),1)

PGM        = i.hyper.atcorr
PREFIX    ?= /usr/local
BINDIR     = $(DESTDIR)$(PREFIX)/bin

CC        ?= cc
SIXSV_CFLAGS  := $(shell pkg-config --cflags libsixsv  2>/dev/null || echo "-I/usr/include/sixsv")
SIXSV_LIBS    := $(shell pkg-config --libs   libsixsv  2>/dev/null || echo "-lsixsv")
RAS3D_CFLAGS  := $(shell pkg-config --cflags ras3d     2>/dev/null || echo "-I/usr/include -DHAVE_RAS3D")
RAS3D_LIBS    := $(shell pkg-config --libs   ras3d     2>/dev/null || echo "-lras3d")

CFLAGS  ?= -O3 -ffast-math
CFLAGS  += -std=c11 -Wall -Wextra -Wno-unused-parameter -fopenmp \
            $(SIXSV_CFLAGS) $(RAS3D_CFLAGS)

# On Debian trixie the serial HDF5 library is libhdf5_serial; fall back to
# plain -lhdf5 on other distros where it is linked differently.
HDF5_LIBS := $(shell \
    ldconfig -p 2>/dev/null | grep -q libhdf5_serial && echo "-lhdf5_serial" \
    || pkg-config --libs hdf5 2>/dev/null \
    || echo "-lhdf5")

LDFLAGS += -fopenmp $(SIXSV_LIBS) $(RAS3D_LIBS) \
            -lgeotiff -ltiff $(HDF5_LIBS) -lm

$(PGM): main.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

install: $(PGM)
	install -d $(BINDIR)
	install -m 755 $(PGM) $(BINDIR)/$(PGM)

clean:
	rm -f $(PGM)

.PHONY: install clean

# ── GRASS Module.make build (default) ─────────────────────────────────────────
else

MODULE_TOPDIR = $(HOME)/dev/grass

PGM = i.hyper.atcorr

SIXSV_LIB_NAME = grass_sixsv.$(GRASS_LIB_VERSION_NUMBER)
SIXSVLIB        = -l$(SIXSV_LIB_NAME)
SIXSVDEP        = $(ARCH_LIBDIR)/$(SHLIB_PREFIX)$(SIXSV_LIB_NAME)$(SHLIB_SUFFIX)

MOD_OBJS = main.o

EXTRA_INC     = -I$(HOME)/dev/libsixsv/include
EXTRA_CFLAGS  = -O3 -ffast-math $(OPENMP_CFLAGS) -std=c11 \
                -Wall -Wextra -Wno-unused-parameter
EXTRA_LDFLAGS = $(OPENMP_LIBPATH) $(OPENMP_LIB) -lgomp
LIBES         = $(SIXSVLIB) $(RASTER3DLIB) $(RASTERLIB) $(GISLIB)
DEPENDENCIES  = $(SIXSVDEP)

include $(MODULE_TOPDIR)/include/Make/Module.make

default: cmd

install:
	$(INSTALL) $(ARCH_DISTDIR)/bin/$(PGM)$(EXE) $(INST_DIR)/bin/
	$(INSTALL_DATA) $(HTMLDIR)/$(PGM).html $(INST_DIR)/docs/html/
	$(INSTALL_DATA) $(ARCH_DISTDIR)/docs/man/man1/$(PGM).1 \
	    $(INST_DIR)/docs/man/man1/ 2>/dev/null || true

doxygen:
	doxygen Doxyfile
	@echo "Programmer's manual: doc/html/index.html"

.PHONY: install doxygen

endif
