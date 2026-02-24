#!/usr/bin/env python3
"""
Extract Fortran DATA arrays from 6SV2.1 source files and generate C static tables.
Produces:
  - src/gas_tables.c + include/gas_tables.h
  - src/aerosol_tables.c + include/aerosol_tables.h
  - src/solar_table.c (SOLIRR 1501-point array)
"""
import re
import sys
import os

SIXSV_DIR = os.path.expanduser("~/dev/6sV2.1")
OUT_SRC = os.path.join(os.path.dirname(__file__), "../src")
OUT_INC = os.path.join(os.path.dirname(__file__), "../include")


def fortran_float(s):
    """Convert Fortran float literal to Python float."""
    s = s.strip().replace('E', 'e').replace('D', 'e')
    return float(s)


def extract_data_array(filename):
    """
    Parse a Fortran file and extract all values from DATA statements.
    Returns a flat list of floats in order.
    """
    path = os.path.join(SIXSV_DIR, filename)
    with open(path) as f:
        lines = f.readlines()

    values = []
    in_data = False
    continuation_buf = ""

    for line in lines:
        # Strip line number prefix and continuation character
        if len(line) > 6:
            cont_char = line[5]
            content = line[6:].rstrip('\n')
        else:
            continue

        if not in_data:
            # Look for DATA statement
            if re.match(r'\s+DATA\s', content, re.IGNORECASE) or \
               re.match(r'\s+data\s', content, re.IGNORECASE):
                in_data = True
                continuation_buf = content
        else:
            if cont_char in ('*', 'a', 'A', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', 's'):
                continuation_buf += " " + content
            else:
                # Process previous data statement
                _extract_values_from_data(continuation_buf, values)
                if re.match(r'\s+DATA\s', content, re.IGNORECASE) or \
                   re.match(r'\s+data\s', content, re.IGNORECASE):
                    continuation_buf = content
                else:
                    in_data = False
                    continuation_buf = ""

    if in_data and continuation_buf:
        _extract_values_from_data(continuation_buf, values)

    return values


def _extract_values_from_data(text, values):
    """Extract numeric values from a Fortran DATA statement text."""
    # Remove everything before the /
    parts = text.split('/')
    if len(parts) < 2:
        return
    data_content = parts[1]

    # Find all floats/ints
    tokens = re.findall(r'[+-]?\d+\.?\d*[eEdD]?[+-]?\d*', data_content)
    for tok in tokens:
        try:
            values.append(fortran_float(tok))
        except Exception:
            pass


def extract_2d_array_ordered(filename, nrows, ncols):
    """
    Extract a 2D array from Fortran DATA statements of the form:
       DATA ((ACR(K,J),K=1,ncols),J=j1,j2) / ... /
    Returns a 2D list [row][col] (0-indexed).
    """
    path = os.path.join(SIXSV_DIR, filename)
    with open(path) as f:
        content = f.read()

    # Extract all data blocks
    # Pattern: DATA ((ACR(K,J),K=1,N),J=j1,j2) / values /
    result = [[0.0] * ncols for _ in range(nrows)]

    # Concatenate continuation lines
    # Replace line numbers + continuation
    cleaned_lines = []
    for line in content.split('\n'):
        if len(line) >= 7:
            cont = line[5]
            body = line[6:]
            if cont in ('a', 'A', '*', 's', '+') or (cont.isdigit() and cont != ' '):
                if cleaned_lines:
                    cleaned_lines[-1] += ' ' + body.strip()
                    continue
            cleaned_lines.append(body)
        elif line.strip():
            cleaned_lines.append(line)

    # Now process each line
    j_current = 0
    k_current = 0

    full_text = ' '.join(cleaned_lines)
    # Find all DATA blocks
    data_blocks = re.findall(
        r'DATA\s*\(\s*\(\s*\w+\s*\(\s*K\s*,\s*J\s*\)\s*,\s*K\s*=\s*1\s*,\s*\d+\s*\)\s*,\s*J\s*=\s*(\d+)\s*,\s*(\d+)\s*\)\s*/([^/]+)/',
        full_text, re.IGNORECASE
    )

    if not data_blocks:
        # Try simpler pattern
        data_blocks = re.findall(
            r'DATA\s*\(\s*\(\s*\w+\s*\([^)]+\)[^)]+\)\s*,\s*J\s*=\s*(\d+)\s*,\s*(\d+)\s*\)\s*/([^/]+)/',
            full_text, re.IGNORECASE
        )

    for j1_str, j2_str, vals_str in data_blocks:
        j1, j2 = int(j1_str), int(j2_str)
        tokens = re.findall(r'[+-]?\d+\.?\d*[eEdD]?[+-]?\d*', vals_str)
        floats = [fortran_float(t) for t in tokens]
        idx = 0
        for j in range(j1, j2 + 1):
            for k in range(1, ncols + 1):
                if idx < len(floats):
                    result[j - 1][k - 1] = floats[idx]
                    idx += 1

    return result


# ─── Gas tables extraction ─────────────────────────────────────────────────────

GAS_FILES = {
    # (gas_id, band_id): filename, n_intervals
    (1, 1): ('WAVA1.f', 256),
    (1, 2): ('WAVA2.f', 256),
    (1, 3): ('WAVA3.f', 256),
    (1, 4): ('WAVA4.f', 256),
    (1, 5): ('WAVA5.f', 256),
    (1, 6): ('WAVA6.f', 256),
    (2, 1): ('DICA1.f', 256),
    (2, 2): ('DICA2.f', 256),
    (2, 3): ('DICA3.f', 256),
    (3, 1): ('OZON1.f', 102),   # special: ozone uses co3[102], handled separately
    (3, 3): ('OXYG3.f', 256),
    (3, 4): ('OXYG4.f', 256),
    (3, 5): ('OXYG5.f', 256),
    (3, 6): ('OXYG6.f', 256),
    (5, 1): ('NIOX1.f', 256),
    (5, 2): ('NIOX2.f', 256),
    (5, 3): ('NIOX3.f', 256),
    (5, 4): ('NIOX4.f', 256),
    (5, 5): ('NIOX5.f', 256),
    (5, 6): ('NIOX6.f', 256),
    (6, 1): ('METH1.f', 256),
    (6, 2): ('METH2.f', 256),
    (6, 3): ('METH3.f', 256),
    (6, 4): ('METH4.f', 256),
    (6, 5): ('METH5.f', 256),
    (6, 6): ('METH6.f', 256),
    (7, 1): ('MOCA1.f', 256),
    (7, 2): ('MOCA2.f', 256),
    (7, 3): ('MOCA3.f', 256),
    (7, 4): ('MOCA4.f', 256),
    (7, 5): ('MOCA5.f', 256),
    (7, 6): ('MOCA6.f', 256),
}

GAS_NAMES = {1: 'wava', 2: 'dica', 3: 'oxyg', 4: 'ozon', 5: 'niox', 6: 'meth', 7: 'moca'}


def extract_acr_array(filename, n_intervals):
    """Extract the ACR(8, n) array from a WAVA/OXYG/etc file."""
    path = os.path.join(SIXSV_DIR, filename)
    with open(path) as f:
        raw = f.read()

    # Join continuation lines
    joined = []
    prev = None
    for line in raw.split('\n'):
        if len(line) < 6:
            if prev is not None:
                joined.append(prev)
            prev = line
            continue
        cont = line[5] if len(line) > 5 else ' '
        body = line[6:] if len(line) > 6 else ''
        if cont not in (' ', '!', 'C', 'c') and not body.strip().upper().startswith('SUBROUTINE') \
                and not body.strip().upper().startswith('END') \
                and not body.strip().upper().startswith('REAL') \
                and not body.strip().upper().startswith('INTEGER') \
                and not body.strip().upper().startswith('DATA') \
                and cont.upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+*':
            if joined:
                joined[-1] += ' ' + body.strip()
            elif prev:
                prev += ' ' + body.strip()
        else:
            if prev is not None:
                joined.append(prev)
            prev = body

    if prev:
        joined.append(prev)

    full = ' '.join(joined)

    # Find all DATA blocks and extract values grouped by interval
    result = [[0.0] * 8 for _ in range(n_intervals)]

    # Pattern: DATA ((ACR(K,J),K=1,8),J=j1,j2) / values /
    for m in re.finditer(
        r'DATA\s*\(\s*\(\s*ACR\s*\(\s*K\s*,\s*J\s*\)\s*,\s*K\s*=\s*1\s*,\s*8\s*\)\s*,\s*J\s*=\s*(\d+)\s*,\s*(\d+)\s*\)\s*/([^/]+)/',
        full, re.IGNORECASE
    ):
        j1, j2 = int(m.group(1)), int(m.group(2))
        vals_str = m.group(3)
        tokens = re.findall(r'[+-]?\d+\.?\d*[eEdD]?[+-]?\d*', vals_str)
        floats = [fortran_float(t) for t in tokens]
        idx = 0
        for j in range(j1, j2 + 1):
            for k in range(8):
                if idx < len(floats) and j - 1 < n_intervals:
                    result[j - 1][k] = floats[idx]
                idx += 1

    return result


def write_gas_tables():
    """Generate src/gas_tables.c and include/gas_tables.h"""
    header_lines = [
        "/* Auto-generated by tools/extract_tables.py from 6SV2.1 Fortran source */",
        "#pragma once",
        "#include <stddef.h>",
        "",
        "/* Gas absorption coefficient tables from 6SV2.1.",
        " * For gas g, spectral band b (1-6), interval i (0-255):",
        " *   gas_acr[g][b][i][0..7] = 8 spectral parameters",
        " * Indexing: gas: 1=H2O, 2=CO2, 3=O2, 5=N2O, 6=CH4, 7=CO (4=O3 special)",
        " */",
        "",
        "/* Maximum intervals per band */",
        "#define GAS_MAX_INTERVALS 256",
        "#define GAS_NBANDS 6",
        "",
        "/* Flat table per gas/band: gas_table_wava[band-1][interval][coef] */",
    ]

    src_lines = [
        "/* Auto-generated by tools/extract_tables.py from 6SV2.1 Fortran source */",
        '#include "gas_tables.h"',
        "",
    ]

    gas_ids = [1, 2, 3, 5, 6, 7]
    gas_short = {1: 'wava', 2: 'dica', 3: 'oxyg', 5: 'niox', 6: 'meth', 7: 'moca'}
    band_ranges = {1: 1, 2: 2, 3: 2, 5: 6, 6: 6, 7: 6}  # first band for each gas
    # Actually: which bands are valid for each gas (from ABSTRA dispatch table):
    # H2O (1): bands 1-6
    # CO2 (2): bands 1,2,3
    # O2  (3): bands 3,4,5,6
    # O3  (4): special (co3 array, not acr)
    # N2O (5): bands 1-6
    # CH4 (6): bands 1-6
    # CO  (7): bands 1-6
    gas_bands = {
        1: [1, 2, 3, 4, 5, 6],
        2: [1, 2, 3],
        3: [3, 4, 5, 6],
        5: [1, 2, 3, 4, 5, 6],
        6: [1, 2, 3, 4, 5, 6],
        7: [1, 2, 3, 4, 5, 6],
    }

    # Per-gas arrays
    for gid in gas_ids:
        gname = gas_short[gid]
        bands = gas_bands[gid]
        n_bands = 6  # always declare 6 bands (some will be zeros)

        header_lines.append(f"extern const float gas_acr_{gname}[6][256][8];  /* gas_id={gid} */")

        # Generate C array
        arr_lines = [f"const float gas_acr_{gname}[6][256][8] = {{"]
        for b in range(1, 7):
            arr_lines.append(f"  /* band {b} */")
            arr_lines.append("  {")
            if b in bands:
                fname = f"{gname.upper()}{b}.f" if (gid, b) not in {(2,1):''}.items() else f"DICA{b}.f"
                # Correct filenames
                fmap = {
                    (1, 1): 'WAVA1.f', (1, 2): 'WAVA2.f', (1, 3): 'WAVA3.f',
                    (1, 4): 'WAVA4.f', (1, 5): 'WAVA5.f', (1, 6): 'WAVA6.f',
                    (2, 1): 'DICA1.f', (2, 2): 'DICA2.f', (2, 3): 'DICA3.f',
                    (3, 3): 'OXYG3.f', (3, 4): 'OXYG4.f', (3, 5): 'OXYG5.f', (3, 6): 'OXYG6.f',
                    (5, 1): 'NIOX1.f', (5, 2): 'NIOX2.f', (5, 3): 'NIOX3.f',
                    (5, 4): 'NIOX4.f', (5, 5): 'NIOX5.f', (5, 6): 'NIOX6.f',
                    (6, 1): 'METH1.f', (6, 2): 'METH2.f', (6, 3): 'METH3.f',
                    (6, 4): 'METH4.f', (6, 5): 'METH5.f', (6, 6): 'METH6.f',
                    (7, 1): 'MOCA1.f', (7, 2): 'MOCA2.f', (7, 3): 'MOCA3.f',
                    (7, 4): 'MOCA4.f', (7, 5): 'MOCA5.f', (7, 6): 'MOCA6.f',
                }
                fname = fmap.get((gid, b), None)
                if fname and os.path.exists(os.path.join(SIXSV_DIR, fname)):
                    print(f"  Extracting {fname}...", flush=True)
                    acr = extract_acr_array(fname, 256)
                    for i, row in enumerate(acr):
                        vals = ', '.join(f'{v:.7E}f' for v in row)
                        comma = ',' if i < 255 else ''
                        arr_lines.append(f"    {{ {vals} }}{comma}")
                else:
                    for i in range(256):
                        comma = ',' if i < 255 else ''
                        arr_lines.append(f"    {{ 0,0,0,0,0,0,0,0 }}{comma}")
            else:
                for i in range(256):
                    comma = ',' if i < 255 else ''
                    arr_lines.append(f"    {{ 0,0,0,0,0,0,0,0 }}{comma}")
            arr_lines.append("  }," if b < 6 else "  }")
        arr_lines.append("};")
        arr_lines.append("")
        src_lines.extend(arr_lines)

    # Ozone table (special: co3[102] from ABSTRA.f)
    ozon_data = [
        4.50e-03, 8.00e-03, 1.07e-02, 1.10e-02, 1.27e-02, 1.71e-02,
        2.00e-02, 2.45e-02, 3.07e-02, 3.84e-02, 4.78e-02, 5.67e-02,
        6.54e-02, 7.62e-02, 9.15e-02, 1.00e-01, 1.09e-01, 1.20e-01,
        1.28e-01, 1.12e-01, 1.11e-01, 1.16e-01, 1.19e-01, 1.13e-01,
        1.03e-01, 9.24e-02, 8.28e-02, 7.57e-02, 7.07e-02, 6.58e-02,
        5.56e-02, 4.77e-02, 4.06e-02, 3.87e-02, 3.82e-02, 2.94e-02,
        2.09e-02, 1.80e-02, 1.91e-02, 1.66e-02, 1.17e-02, 7.70e-03,
        6.10e-03, 8.50e-03, 6.10e-03, 3.70e-03, 3.20e-03, 3.10e-03,
        2.55e-03, 1.98e-03, 1.40e-03, 8.25e-04, 2.50e-04, 0.,
        0., 0., 5.65e-04, 2.04e-03, 7.35e-03, 2.03e-02,
        4.98e-02, 1.18e-01, 2.46e-01, 5.18e-01, 1.02e+00, 1.95e+00,
        3.79e+00, 6.65e+00, 1.24e+01, 2.20e+01, 3.67e+01, 5.95e+01,
        8.50e+01, 1.26e+02, 1.68e+02, 2.06e+02, 2.42e+02, 2.71e+02,
        2.91e+02, 3.02e+02, 3.03e+02, 2.94e+02, 2.77e+02, 2.54e+02,
        2.26e+02, 1.96e+02, 1.68e+02, 1.44e+02, 1.17e+02, 9.75e+01,
        7.65e+01, 6.04e+01, 4.62e+01, 3.46e+01, 2.52e+01, 2.00e+01,
        1.57e+01, 1.20e+01, 1.00e+01, 8.80e+00, 8.30e+00, 8.60e+00
    ]
    header_lines.append("extern const float gas_co3_ozon[102];  /* ozone abs cross-section (13000-27500 cm-1) */")
    src_lines.append("const float gas_co3_ozon[102] = {")
    for i in range(0, 102, 6):
        chunk = ozon_data[i:i+6]
        vals = ', '.join(f'{v:.7E}f' for v in chunk)
        src_lines.append(f"  {vals},")
    src_lines.append("};")
    src_lines.append("")

    # CCH2O continuum absorption coefficients
    cch2o = [0.00, 0.19, 0.15, 0.12, 0.10, 0.09, 0.10, 0.12, 0.15, 0.17, 0.20, 0.24, 0.28, 0.33, 0.00]
    header_lines.append("extern const float gas_cch2o[15];  /* H2O continuum (2350-3000 cm-1) */")
    src_lines.append("const float gas_cch2o[15] = {")
    src_lines.append("  " + ", ".join(f"{v:.2f}f" for v in cch2o))
    src_lines.append("};")
    src_lines.append("")

    os.makedirs(OUT_SRC, exist_ok=True)
    os.makedirs(OUT_INC, exist_ok=True)

    with open(os.path.join(OUT_INC, "gas_tables.h"), "w") as f:
        f.write("\n".join(header_lines) + "\n")

    with open(os.path.join(OUT_SRC, "gas_tables.c"), "w") as f:
        f.write("\n".join(src_lines) + "\n")

    print("  → gas_tables.h and gas_tables.c written")


# ─── Aerosol tables extraction ─────────────────────────────────────────────────

def extract_aerosol_component(filename):
    """
    Extract from DUST.f / WATE.f / OCEA.f / SOOT.f:
    - ex_m(20), sc_m(20), asy_m(20): extinction, scattering, asymmetry at 20 wavelengths
    - phr(20, 83): phase function at 20 wavelengths x 83 Gauss points
    Returns dict with 'ext', 'sca', 'asy', 'pha' arrays.
    """
    path = os.path.join(SIXSV_DIR, filename)
    with open(path) as f:
        raw = f.read()

    # Join continuation lines
    joined = []
    for line in raw.split('\n'):
        if len(line) < 6:
            joined.append(line)
            continue
        cont = line[5]
        body = line[6:] if len(line) > 6 else ''
        if cont not in (' ', '!', 'C', 'c', '\t') and cont != '0':
            if joined:
                joined[-1] += ' ' + body.strip()
            else:
                joined.append(body)
        else:
            joined.append(body)

    full = ' '.join(joined)

    def get_simple_array(name, n):
        """Extract 1D array from DATA (name(j),j=1,n) / ... / """
        patterns = [
            rf'data\s*\(\s*{name}\s*\(\s*j\s*\)\s*,\s*j\s*=\s*1\s*,\s*{n}\s*\)\s*/([^/]+)/',
            rf'DATA\s*\(\s*{name}\s*\(\s*j\s*\)\s*,\s*j\s*=\s*1\s*,\s*{n}\s*\)\s*/([^/]+)/',
        ]
        for pat in patterns:
            m = re.search(pat, full, re.IGNORECASE)
            if m:
                tokens = re.findall(r'[+-]?\d+\.?\d*[eEdD]?[+-]?\d*', m.group(1))
                return [fortran_float(t) for t in tokens[:n]]
        return [0.0] * n

    def get_paired_array(name1, name2, n):
        """Extract paired array DATA (n1(j),n2(j),j=1,n) / ..."""
        m = re.search(
            rf'data\s*\(\s*{name1}\s*\(\s*j\s*\)\s*,\s*{name2}\s*\(\s*j\s*\)\s*,\s*j\s*=\s*1\s*,\s*{n}\s*\)\s*/([^/]+)/',
            full, re.IGNORECASE
        )
        if m:
            tokens = re.findall(r'[+-]?\d+\.?\d*[eEdD]?[+-]?\d*', m.group(1))
            floats = [fortran_float(t) for t in tokens]
            a1, a2 = [], []
            for k in range(n):
                a1.append(floats[2*k] if 2*k < len(floats) else 0.0)
                a2.append(floats[2*k+1] if 2*k+1 < len(floats) else 0.0)
            return a1, a2
        return [0.0]*n, [0.0]*n

    asy = get_simple_array('asy_m', 20)
    ext, sca = get_paired_array('ex_m', 'sc_m', 20)

    # Phase function: DATA ((PHR(i,j),j=1,83),i=i1,i2) / ... /
    pha = [[0.0] * 83 for _ in range(20)]
    for m in re.finditer(
        r'DATA\s*\(\s*\(\s*PHR\s*\(\s*i\s*,\s*j\s*\)\s*,\s*j\s*=\s*1\s*,\s*83\s*\)\s*,\s*i\s*=\s*(\d+)\s*,\s*(\d+)\s*\)\s*/([^/]+)/',
        full, re.IGNORECASE
    ):
        i1, i2 = int(m.group(1)), int(m.group(2))
        tokens = re.findall(r'[+-]?\d+\.?\d*[eEdD]?[+-]?\d*', m.group(3))
        floats = [fortran_float(t) for t in tokens]
        idx = 0
        for i in range(i1, i2 + 1):
            for j in range(83):
                if idx < len(floats):
                    pha[i - 1][j] = floats[idx]
                idx += 1

    return {'ext': ext, 'sca': sca, 'asy': asy, 'pha': pha}


def write_aerosol_tables():
    """Generate src/aerosol_tables.c and include/aerosol_tables.h"""
    components = ['dust', 'wate', 'ocea', 'soot']
    filenames = ['DUST.f', 'WATE.f', 'OCEA.f', 'SOOT.f']

    header_lines = [
        "/* Auto-generated by tools/extract_tables.py from 6SV2.1 Fortran source */",
        "#pragma once",
        "",
        "/* Aerosol component optical properties at 20 reference wavelengths (wldis).",
        " * Components: 0=dust, 1=wate(water-soluble), 2=ocea(oceanic), 3=soot",
        " * For standard aerosol models (continental/maritime/urban/desert):",
        " *   each model is a mixture of these 4 components.",
        " */",
        "",
        "#define AEROSOL_NCOMP 4",
        "#define AEROSOL_NWL   20",
        "#define AEROSOL_NQUAD 83",
        "",
    ]
    src_lines = [
        "/* Auto-generated by tools/extract_tables.py from 6SV2.1 Fortran source */",
        '#include "aerosol_tables.h"',
        "",
    ]

    for comp, fname in zip(components, filenames):
        print(f"  Extracting {fname}...", flush=True)
        d = extract_aerosol_component(fname)

        header_lines.append(f"extern const float aerosol_{comp}_ext[20];")
        header_lines.append(f"extern const float aerosol_{comp}_sca[20];")
        header_lines.append(f"extern const float aerosol_{comp}_asy[20];")
        header_lines.append(f"extern const float aerosol_{comp}_pha[20][83];")
        header_lines.append("")

        # ext
        src_lines.append(f"const float aerosol_{comp}_ext[20] = {{")
        src_lines.append("  " + ", ".join(f"{v:.7E}f" for v in d['ext']))
        src_lines.append("};")
        # sca
        src_lines.append(f"const float aerosol_{comp}_sca[20] = {{")
        src_lines.append("  " + ", ".join(f"{v:.7E}f" for v in d['sca']))
        src_lines.append("};")
        # asy
        src_lines.append(f"const float aerosol_{comp}_asy[20] = {{")
        src_lines.append("  " + ", ".join(f"{v:.7E}f" for v in d['asy']))
        src_lines.append("};")
        # pha
        src_lines.append(f"const float aerosol_{comp}_pha[20][83] = {{")
        for i, row in enumerate(d['pha']):
            vals = ", ".join(f"{v:.7E}f" for v in row)
            comma = "," if i < 19 else ""
            src_lines.append(f"  {{ {vals} }}{comma}")
        src_lines.append("};")
        src_lines.append("")

    # Standard mixing ratios (from 6SV2.1 main.f)
    # ci(1..4) = [dust, wate, ocea, soot] volume fractions
    header_lines.append("/* Standard aerosol model mixing ratios [dust, wate, ocea, soot] */")
    header_lines.append("/* Models: 0=continental, 1=maritime, 2=urban */")
    header_lines.append("extern const float aerosol_std_mix[3][4];")

    src_lines.append("/* Standard aerosol model mixing ratios [dust, wate, ocea, soot] */")
    src_lines.append("/* Models: 0=continental, 1=maritime, 2=urban */")
    src_lines.append("const float aerosol_std_mix[3][4] = {")
    src_lines.append("  { 0.70f, 0.29f, 0.00f, 0.01f },  /* continental */")
    src_lines.append("  { 0.05f, 0.95f, 0.00f, 0.00f },  /* maritime */")
    src_lines.append("  { 0.21f, 0.30f, 0.00f, 0.49f },  /* urban */")
    src_lines.append("};")
    src_lines.append("")

    with open(os.path.join(OUT_INC, "aerosol_tables.h"), "w") as f:
        f.write("\n".join(header_lines) + "\n")
    with open(os.path.join(OUT_SRC, "aerosol_tables.c"), "w") as f:
        f.write("\n".join(src_lines) + "\n")

    print("  → aerosol_tables.h and aerosol_tables.c written")


# ─── Solar irradiance table ─────────────────────────────────────────────────────

def write_solar_table():
    """Extract SOLIRR 1501-point table from SOLIRR.f."""
    si_data = [
        69.30, 77.65, 86.00, 100.06, 114.12, 137.06, 160.00,
        169.52, 179.04, 178.02, 177.00, 193.69, 210.38, 241.69,
        273.00, 318.42, 363.84, 434.42, 505.00, 531.50, 558.00,
        547.50, 537.00, 559.02, 581.03, 619.52, 658.00, 694.39,
        730.78, 774.39, 817.99, 871.99, 925.99, 912.04, 898.09,
        920.69, 943.29, 925.99, 908.69, 936.09, 963.49, 994.94,
        1026.39, 980.74, 935.09, 1036.29, 1137.49, 1163.74, 1189.99,
        1109.34, 1028.69, 1088.99, 1149.29, 1033.69, 918.09, 1031.89,
        1145.69, 1035.09, 924.49, 1269.29, 1614.09, 1631.09, 1648.09,
        1677.19, 1706.29, 1744.89, 1783.49, 1750.19, 1716.89, 1705.19,
        1693.49, 1597.69, 1501.89, 1630.99, 1760.09, 1775.24, 1790.39,
        1859.94, 1929.49, 1993.44, 2057.39, 2039.23, 2021.08, 2030.73,
        2040.38, 2026.53, 2012.68, 1999.53, 1986.38, 2002.88, 2019.38,
        2038.09, 2056.79, 1967.74, 1878.68, 1905.83, 1932.98, 1953.58,
        1974.18, 1935.68, 1897.19, 1916.78, 1936.38, 1937.23, 1938.09,
        1881.44, 1824.79, 1814.09, 1803.39, 1832.24, 1861.09, 1885.93,
        1910.78, 1904.68, 1898.58, 1875.73, 1852.88, 1865.64, 1878.39,
        1874.74, 1871.09, 1872.44, 1873.79, 1850.39, 1826.99, 1837.04,
        1847.09, 1841.18, 1835.28, 1849.48, 1863.69, 1851.03, 1838.38,
        1840.73, 1843.08, 1802.83, 1762.58, 1778.78, 1794.99, 1777.48,
        1759.98, 1764.73, 1769.49, 1753.48, 1737.48, 1713.14, 1688.80,
        1702.88, 1716.97, 1696.07, 1675.17, 1672.03, 1668.89, 1663.56,
        1658.23, 1647.75, 1637.27, 1630.02, 1622.77, 1606.06, 1589.36,
        1552.29, 1515.22, 1528.91, 1542.60, 1548.90, 1555.21, 1544.41,
        1533.62, 1525.24, 1516.86, 1507.92, 1498.98, 1484.07, 1469.17,
        1464.28, 1459.39, 1448.73, 1438.08, 1423.16, 1408.24, 1407.53,
        1406.82, 1397.82, 1388.82, 1378.51, 1368.21, 1352.13, 1336.05,
        1343.88, 1351.71, 1339.60, 1327.50, 1320.72, 1313.94, 1294.94,
        1275.94, 1280.92, 1285.90, 1278.04, 1270.19, 1263.68, 1257.18,
        1249.80, 1242.41, 1231.30, 1220.19, 1212.14, 1204.10, 1201.69,
        1199.29, 1194.78, 1190.27, 1185.47, 1180.68, 1174.38, 1168.09,
        1156.17, 1144.26, 1143.46, 1142.67, 1132.95, 1123.23, 1116.71,
        1110.19, 1110.89, 1111.59, 1094.80, 1078.01, 1077.75, 1077.49,
        1073.89, 1070.29, 1058.71, 1047.13, 1045.66, 1044.20, 1037.03,
        1029.86, 1010.40, 990.94, 966.91, 942.89, 972.87, 1002.86,
        978.93, 955.00, 960.95, 966.91, 983.31, 999.71, 991.91,
        984.11, 979.05, 973.99, 968.79, 963.60, 958.23, 952.87,
        947.93, 942.99, 937.99, 933.00, 928.00, 923.00, 918.18,
        913.37, 908.74, 904.11, 899.05, 893.99, 889.18, 884.37,
        879.74, 875.12, 870.24, 865.36, 860.94, 856.53, 852.02,
        847.50, 843.00, 838.50, 833.99, 829.49, 824.98, 820.48,
        815.99, 811.50, 806.99, 802.49, 798.17, 793.86, 789.74,
        785.63, 781.25, 776.87, 772.92, 768.98, 764.80, 760.63,
        756.06, 751.49, 746.99, 742.49, 738.18, 733.88, 729.76,
        725.63, 721.24, 716.86, 712.92, 708.99, 704.81, 700.63,
        696.25, 691.87, 687.94, 684.01, 680.01, 676.00, 671.80,
        667.61, 663.23, 658.86, 655.32, 651.77, 649.07, 646.37,
        643.74, 641.11, 638.05, 634.99, 632.18, 629.37, 626.74,
        624.12, 621.06, 618.00, 615.18, 612.37, 609.92, 607.48,
        604.79, 602.11, 599.24, 596.38, 593.93, 591.48, 588.79,
        586.11, 583.25, 580.40, 577.94, 575.48, 572.99, 570.51,
        568.00, 565.49, 562.98, 560.47, 557.98, 555.50, 553.01,
        550.51, 548.00, 545.49, 542.98, 540.48, 537.98, 535.49,
        533.19, 530.90, 528.94, 526.99, 524.80, 522.62, 520.24,
        517.87, 515.44, 513.01, 509.59, 506.17, 502.89, 499.62,
        496.35, 493.09, 489.81, 486.54, 483.27, 480.01, 476.73,
        473.46, 470.19, 466.92, 463.64, 460.37, 457.10, 453.84,
        450.57, 447.30, 444.03, 440.76, 437.48, 434.21, 430.94,
        427.67, 424.40, 421.13, 417.86, 414.59, 411.32, 408.05,
        404.78, 401.51, 398.24, 394.97, 391.70, 388.43, 392.57,
        396.71, 401.92, 407.14, 405.32, 403.50, 401.67, 399.84,
        398.02, 396.21, 394.37, 392.54, 390.72, 388.90, 387.06,
        385.23, 383.42, 381.60, 379.77, 377.95, 376.12, 374.30,
        372.48, 370.66, 368.82, 366.99, 365.17, 363.35, 361.52,
        359.69, 357.87, 356.05, 354.22, 352.39, 350.57, 348.75,
        346.92, 345.10, 343.27, 341.45, 341.84, 342.24, 342.95,
        343.66, 342.27, 340.89, 339.49, 338.09, 336.69, 335.30,
        333.91, 332.53, 331.13, 329.73, 328.34, 326.96, 325.56,
        324.16, 322.77, 321.39, 319.99, 318.59, 317.20, 315.82,
        314.42, 313.03, 311.63, 310.24, 308.85, 307.46, 306.06,
        304.66, 303.28, 301.90, 300.50, 299.10, 297.71, 296.32,
        294.93, 293.54, 293.41, 293.28, 293.35, 293.42, 292.26,
        291.10, 289.97, 288.84, 287.69, 286.54, 285.39, 284.25,
        283.10, 281.96, 280.81, 279.67, 278.52, 277.38, 276.23,
        275.08, 273.94, 272.80, 271.65, 270.51, 269.36, 268.22,
        267.07, 265.93, 264.78, 263.64, 262.49, 261.34, 260.20,
        259.06, 257.91, 256.77, 255.62, 254.47, 253.33, 252.20,
        251.16, 250.13, 249.11, 248.09, 246.97, 245.86, 244.74,
        243.61, 242.49, 241.37, 240.24, 239.12, 238.00, 236.89,
        235.76, 234.64, 233.51, 232.38, 231.26, 230.13, 229.01,
        227.90, 226.77, 225.65, 224.53, 223.42, 222.29, 221.16,
        220.04, 218.92, 217.80, 216.68, 215.55, 214.43, 213.30,
        212.18, 211.06, 209.94, 208.82, 207.69, 206.99, 206.29,
        205.65, 205.02, 203.98, 202.95, 201.90, 200.85, 199.81,
        198.78, 197.74, 196.70, 195.65, 194.61, 193.57, 192.54,
        191.50, 190.47, 189.42, 188.37, 187.33, 186.30, 185.26,
        184.22, 183.18, 182.14, 181.10, 180.06, 179.02, 177.98,
        176.93, 175.89, 174.86, 173.83, 172.78, 171.73, 170.70,
        169.67, 168.62, 167.57, 167.59, 167.60, 167.76, 167.93,
        167.09, 166.26, 165.42, 164.58, 163.75, 162.92, 162.08,
        161.25, 160.41, 159.58, 158.74, 157.91, 157.07, 156.24,
        155.40, 154.57, 153.73, 152.90, 152.06, 151.23, 150.39,
        149.56, 148.72, 147.89, 147.06, 146.23, 145.39, 144.55,
        143.71, 142.88, 142.05, 141.22, 140.38, 139.54, 138.70,
        137.86, 137.99, 138.11, 138.36, 138.60, 137.94, 137.29,
        136.64, 136.00, 135.35, 134.71, 134.05, 133.39, 132.74,
        132.09, 131.45, 130.81, 130.15, 129.49, 128.84, 128.20,
        127.55, 126.90, 126.25, 125.60, 124.94, 124.29, 123.64,
        123.00, 122.35, 121.70, 121.05, 120.40, 119.74, 119.09,
        118.45, 117.81, 117.15, 116.50, 115.85, 115.19, 115.25,
        115.31, 115.46, 115.62, 115.11, 114.60, 114.09, 113.58,
        113.06, 112.54, 112.03, 111.53, 111.01, 110.50, 109.99,
        109.47, 108.95, 108.44, 107.93, 107.42, 106.92, 106.42,
        105.89, 105.37, 104.85, 104.34, 103.83, 103.33, 102.81,
        102.29, 101.79, 101.29, 100.77, 100.25, 99.74, 99.22,
        98.71, 98.20, 97.69, 97.18, 97.12, 97.07, 97.09,
        97.11, 96.68, 96.26, 95.84, 95.42, 94.99, 94.56,
        94.14, 93.72, 93.31, 92.89, 92.46, 92.03, 91.61,
        91.19, 90.76, 90.34, 89.92, 89.49, 89.07, 88.66,
        88.24, 87.81, 87.39, 86.97, 86.55, 86.12, 85.69,
        85.26, 84.85, 84.43, 84.01, 83.59, 83.17, 82.75,
        82.32, 81.89, 81.89, 81.89, 81.95, 82.02, 81.68,
        81.35, 81.00, 80.65, 80.32, 79.99, 79.64, 79.30,
        78.96, 78.61, 78.27, 77.94, 77.60, 77.26, 76.91,
        76.57, 76.24, 75.90, 75.56, 75.22, 74.88, 74.54,
        74.20, 73.86, 73.52, 73.18, 72.84, 72.50, 72.16,
        71.82, 71.48, 71.14, 70.80, 70.47, 70.13, 69.79,
        69.76, 69.73, 69.76, 69.80, 69.52, 69.24, 68.96,
        68.68, 68.41, 68.14, 67.85, 67.57, 67.29, 67.02,
        66.75, 66.48, 66.19, 65.90, 65.63, 65.36, 65.08,
        64.80, 64.53, 64.25, 63.97, 63.69, 63.41, 63.14,
        62.85, 62.57, 62.30, 62.03, 61.75, 61.47, 61.19,
        60.92, 60.64, 60.36, 60.08, 59.81, 59.80, 59.80,
        59.82, 59.85, 59.63, 59.40, 59.17, 58.95, 58.73,
        58.50, 58.28, 58.06, 57.83, 57.60, 57.37, 57.15,
        56.93, 56.70, 56.48, 56.26, 56.03, 55.79, 55.57,
        55.36, 55.13, 54.90, 54.66, 54.43, 54.22, 54.00,
        53.77, 53.55, 53.32, 53.09, 52.87, 52.65, 52.43,
        52.20, 51.97, 51.75, 51.72, 51.68, 51.67, 51.67,
        51.48, 51.30, 51.11, 50.92, 50.73, 50.55, 50.37,
        50.18, 49.98, 49.79, 49.61, 49.43, 49.23, 49.04,
        48.85, 48.67, 48.48, 48.30, 48.12, 47.93, 47.73,
        47.54, 47.36, 47.18, 46.98, 46.79, 46.60, 46.42,
        46.24, 46.06, 45.87, 45.67, 45.48, 45.30, 45.12,
        44.93, 44.87, 44.82, 44.80, 44.79, 44.62, 44.45,
        44.29, 44.14, 43.98, 43.83, 43.66, 43.49, 43.34,
        43.18, 43.02, 42.86, 42.70, 42.55, 42.38, 42.21,
        42.06, 41.90, 41.74, 41.58, 41.42, 41.26, 41.10,
        40.94, 40.78, 40.62, 40.46, 40.31, 40.14, 39.97,
        39.81, 39.66, 39.50, 39.34, 39.18, 39.03, 38.99,
        38.96, 38.94, 38.92, 38.79, 38.66, 38.52, 38.38,
        38.25, 38.12, 37.99, 37.86, 37.72, 37.58, 37.44,
        37.30, 37.17, 37.05, 36.91, 36.77, 36.64, 36.50,
        36.36, 36.23, 36.09, 35.96, 35.82, 35.69, 35.55,
        35.42, 35.28, 35.15, 35.01, 34.88, 34.75, 34.61,
        34.47, 34.34, 34.20, 34.07, 34.05, 34.03, 34.03,
        34.03, 33.91, 33.79, 33.68, 33.57, 33.46, 33.35,
        33.23, 33.12, 33.01, 32.90, 32.78, 32.67, 32.55,
        32.44, 32.33, 32.23, 32.11, 32.00, 31.89, 31.77,
        31.66, 31.55, 31.43, 31.31, 31.20, 31.10, 30.99,
        30.87, 30.76, 30.66, 30.54, 30.42, 30.31, 30.20,
        30.08, 29.97, 29.93, 29.90, 29.88, 29.87, 29.76,
        29.66, 29.56, 29.46, 29.36, 29.27, 29.17, 29.08,
        28.98, 28.88, 28.77, 28.67, 28.58, 28.49, 28.39,
        28.30, 28.20, 28.10, 28.00, 27.91, 27.81, 27.71,
        27.61, 27.52, 27.41, 27.31, 27.21, 27.12, 27.03,
        26.93, 26.83, 26.74, 26.64, 26.54, 26.44, 26.35,
        26.33, 26.31, 26.29, 26.28, 26.20, 26.12, 26.04,
        25.95, 25.87, 25.79, 25.71, 25.64, 25.54, 25.45,
        25.37, 25.30, 25.21, 25.12, 25.05, 24.98, 24.89,
        24.80, 24.71, 24.63, 24.55, 24.47, 24.39, 24.31,
        24.22, 24.14, 24.05, 23.97, 23.89, 23.81, 23.73,
        23.66, 23.56, 23.47, 23.39, 23.31, 23.28, 23.26,
        23.23, 23.21, 23.13, 23.06, 22.99, 22.92, 22.84,
        22.76, 22.69, 22.63, 22.55, 22.47, 22.41, 22.35,
        22.27, 22.19, 22.11, 22.04, 21.97, 21.90, 21.83,
        21.76, 21.68, 21.60, 21.53, 21.47, 21.39, 21.31,
        21.24, 21.18, 21.11, 21.03, 20.96, 20.89, 20.81,
        20.73, 20.66, 20.60, 20.57, 20.55, 20.54, 20.53,
        20.46, 20.40, 20.34, 20.28, 20.21, 20.14, 20.08,
        20.03, 19.96, 19.90, 19.83, 19.77, 19.71, 19.65,
        19.59, 19.53, 19.46, 19.39, 19.33, 19.27, 19.21,
        19.15, 19.08, 19.02, 18.96, 18.90, 18.84, 18.78,
        18.71, 18.64, 18.58, 18.53, 18.46, 18.40, 18.33,
        18.27, 18.26, 18.25, 18.24, 18.24, 18.19, 18.14,
        18.08, 18.03, 17.98, 17.93, 17.88, 17.83, 17.77,
        17.71, 17.66, 17.62, 17.56, 17.50, 17.45, 17.41,
        17.35, 17.29, 17.25, 17.21, 17.14, 17.08, 17.04,
        17.00, 16.93, 16.87, 16.83, 16.79, 16.72, 16.66,
        16.61, 16.57, 16.51, 16.46, 16.41, 16.36, 16.34,
        16.33, 16.31, 16.30, 16.26, 16.22, 16.17, 16.13,
        16.08, 16.04, 16.00, 15.96, 15.90, 15.84, 15.81,
        15.78, 15.73, 15.68, 15.63, 15.59, 15.55, 15.50,
        15.45, 15.40, 15.36, 15.32, 15.28, 15.24, 15.18,
        15.13, 15.09, 15.05, 15.01, 14.96, 14.91, 14.87,
        14.82, 14.78, 14.73, 14.69, 14.66, 14.64, 14.64,
        14.63, 14.59, 14.55, 14.50, 14.45, 14.41, 14.38,
        14.35, 14.32, 14.26, 14.21, 14.18, 14.15, 14.10,
        14.05, 14.01, 13.98, 13.94, 13.91, 13.86, 13.82,
        13.78, 13.74, 13.70, 13.67, 13.62, 13.58, 13.54,
        13.50, 13.46, 13.43, 13.39, 13.35, 13.30, 13.25,
        13.22, 13.18, 13.17, 13.16, 13.14, 13.12, 13.09,
        13.06, 13.03, 13.00, 12.96, 12.92, 12.89, 12.85,
        12.81, 12.78, 12.74, 12.70, 12.67, 12.65, 12.61,
        12.57, 12.53, 12.50, 12.46, 12.43, 12.39, 12.36,
        12.32, 12.28, 12.25, 12.22, 12.18, 12.15, 12.11,
        12.07, 12.04, 12.01, 11.97, 11.94, 11.90, 11.86,
        11.85, 11.85, 11.85, 11.84, 11.81, 11.78, 11.75,
        11.72, 11.69, 11.66, 11.63, 11.60, 11.58, 11.55,
        11.51, 11.47, 11.45, 11.42, 11.39, 11.36, 11.33,
        11.30, 11.27, 11.24, 11.21, 11.18, 11.15, 11.12,
        11.09, 11.06, 11.03, 11.00, 10.97, 10.94, 10.91,
        10.89, 10.85, 10.82, 10.78, 10.75, 10.73, 10.72,
        10.71, 10.70, 10.67, 10.64, 10.62, 10.59, 10.55,
        10.52, 10.50, 10.47, 10.44, 10.42, 10.39, 10.37,
        10.34, 10.31, 10.28, 10.25, 10.22, 10.20, 10.17,
        10.15, 10.12, 10.10, 10.06, 10.03, 10.00, 9.98,
        9.95, 9.92, 9.89, 9.86, 9.84, 9.82, 9.79,
        9.75, 9.73, 9.71, 9.70, 9.70, 9.70, 9.70,
        9.67, 9.63, 9.61, 9.59, 9.58, 9.56, 9.53,
        9.50, 9.48, 9.45, 9.43, 9.41, 9.39, 9.36,
        9.34, 9.32, 9.30, 9.27, 9.24, 9.22, 9.20,
        9.18, 9.15, 9.13, 9.11, 9.08, 9.06, 9.05,
        9.02, 8.99, 8.96, 8.94, 8.92, 8.90, 8.87,
        8.85, 8.83, 8.81
    ]
    assert len(si_data) == 1501, f"Expected 1501 values, got {len(si_data)}"

    header = [
        "/* Auto-generated from 6SV2.1 SOLIRR.f */",
        "#pragma once",
        "/* Solar irradiance table: 1501 points from 0.25 to 4.0 µm at 0.0025 µm step.",
        " * Units: W/m²/µm  (same as Kurucz). Interpolation: nearest integer index. */",
        "#define SOLAR_TABLE_N 1501",
        "#define SOLAR_TABLE_WL_START 0.250  /* µm */",
        "#define SOLAR_TABLE_STEP    0.0025  /* µm */",
        "extern const float solar_si[1501];",
    ]
    src = [
        "/* Auto-generated from 6SV2.1 SOLIRR.f */",
        '#include "solar_table.h"',
        "",
        "const float solar_si[1501] = {",
    ]
    for i in range(0, 1501, 7):
        chunk = si_data[i:i+7]
        vals = ", ".join(f"{v:.2f}f" for v in chunk)
        src.append(f"  {vals},")
    src.append("};")

    with open(os.path.join(OUT_INC, "solar_table.h"), "w") as f:
        f.write("\n".join(header) + "\n")
    with open(os.path.join(OUT_SRC, "solar_table.c"), "w") as f:
        f.write("\n".join(src) + "\n")
    print("  → solar_table.h and solar_table.c written")


if __name__ == "__main__":
    print("Extracting 6SV2.1 data tables to C source files...")
    print("Gas tables:")
    write_gas_tables()
    print("Aerosol tables:")
    write_aerosol_tables()
    print("Solar table:")
    write_solar_table()
    print("Done.")
