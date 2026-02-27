MODULE_TOPDIR = $(HOME)/dev/grass

PGM = i.hyper.atcorr

# ── Sources ────────────────────────────────────────────────────────────────────
# All library sources live in src/; main.c is the GRASS module entry point.
LIBSRCS := $(sort $(wildcard src/*.c))

# MOD_OBJS lists basenames only — Module.make prefixes $(OBJDIR)/ automatically.
MOD_OBJS = main.o $(notdir $(LIBSRCS:.c=.o))

# ── Compiler/linker options ────────────────────────────────────────────────────
EXTRA_INC     = -Iinclude -Isrc
EXTRA_CFLAGS  = -O3 -march=native -ffast-math -fopenmp -std=c11 -fPIC \
                -Wall -Wextra -Wno-unused-parameter
EXTRA_LDFLAGS = -fopenmp
LIBES         = $(RASTER3DLIB) $(RASTERLIB) $(GISLIB)

include $(MODULE_TOPDIR)/include/Make/Module.make

default: cmd

# ── Compile library sources from src/ ─────────────────────────────────────────
# Module.make's default rule handles *.c in the current directory only;
# this rule covers the src/ subdirectory.
$(OBJDIR)/%.o: src/%.c | $(OBJDIR)
	$(call compiler_c)

# ── Shared library (for Python ctypes bindings) ────────────────────────────────
# Built alongside the module and installed to $(ARCH_LIBDIR).
ATCORR_SHLIB     = $(ARCH_LIBDIR)/libatcorr$(SHLIB_SUFFIX)
# All objects except main.o go into the library.
ATCORR_SHLIB_OBJ = $(filter-out $(OBJDIR)/main.o,$(ARCH_OBJS))

$(ATCORR_SHLIB): $(ATCORR_SHLIB_OBJ) | $(ARCH_LIBDIR)
	$(CC) -shared -fopenmp \
	    -Wl,-soname,libatcorr$(SHLIB_SUFFIX) \
	    -o $@ $^ $(MATHLIB)
	@echo "Built: $@"

# Hook the shared library into the default build target.
cmd: $(ATCORR_SHLIB)

# ── Installation ───────────────────────────────────────────────────────────────
install: $(ATCORR_SHLIB)
	$(INSTALL) $(ARCH_DISTDIR)/bin/$(PGM)$(EXE) $(INST_DIR)/bin/
	$(INSTALL_DATA) $(HTMLDIR)/$(PGM).html $(INST_DIR)/docs/html/
	$(INSTALL_DATA) $(ARCH_DISTDIR)/docs/man/man1/$(PGM).1 \
	    $(INST_DIR)/docs/man/man1/ 2>/dev/null || true
	$(INSTALL) $(ATCORR_SHLIB) $(INST_DIR)/lib/
	@echo "Installed $(PGM) and libatcorr$(SHLIB_SUFFIX)"

# ── Programmer's manual (Doxygen) ──────────────────────────────────────────────
doxygen:
	doxygen Doxyfile
	@echo "Programmer's manual: doc/html/index.html"

.PHONY: cmd install doxygen
