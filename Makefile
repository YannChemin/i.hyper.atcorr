MODULE_TOPDIR = $(HOME)/dev/grass

PGM = i.hyper.atcorr

SIXSV_LIB_NAME = grass_sixsv.$(GRASS_LIB_VERSION_NUMBER)
SIXSVLIB       = -l$(SIXSV_LIB_NAME)
SIXSVDEP       = $(ARCH_LIBDIR)/$(SHLIB_PREFIX)$(SIXSV_LIB_NAME)$(SHLIB_SUFFIX)

# ── Sources ────────────────────────────────────────────────────────────────────
# Only main.c — all RT physics now live in grass_sixsv (../libsixsv/).
MOD_OBJS = main.o

# ── Compiler / linker options ──────────────────────────────────────────────────
# -I../libsixsv/include exposes atcorr.h and brdf.h in the build tree.
# After install both are under $GISBASE/include/grass/ via the standard INC path.
EXTRA_INC     = -I$(HOME)/dev/libsixsv/include
EXTRA_CFLAGS  = -O3 -ffast-math $(OPENMP_CFLAGS) -std=c11 \
                -Wall -Wextra -Wno-unused-parameter
EXTRA_LDFLAGS = $(OPENMP_LIBPATH) $(OPENMP_LIB) -lgomp
LIBES         = $(SIXSVLIB) $(RASTER3DLIB) $(RASTERLIB) $(GISLIB)
DEPENDENCIES  = $(SIXSVDEP)

include $(MODULE_TOPDIR)/include/Make/Module.make

default: cmd

# ── Installation ───────────────────────────────────────────────────────────────
install:
	$(INSTALL) $(ARCH_DISTDIR)/bin/$(PGM)$(EXE) $(INST_DIR)/bin/
	$(INSTALL_DATA) $(HTMLDIR)/$(PGM).html $(INST_DIR)/docs/html/
	$(INSTALL_DATA) $(ARCH_DISTDIR)/docs/man/man1/$(PGM).1 \
	    $(INST_DIR)/docs/man/man1/ 2>/dev/null || true

# ── Programmer's manual (Doxygen) ──────────────────────────────────────────────
doxygen:
	doxygen Doxyfile
	@echo "Programmer's manual: doc/html/index.html"

.PHONY: install doxygen
