"""
Run on the Windows PC to locate a GenTL .cti producer (needed by
gige_capture.py / Harvesters). Uses only the standard library.

Run:
    python windows_find_cti.py
"""

import os

SEARCH_ROOTS = [
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    r"C:\ProgramData",
]


def find_cti_files():
    found = []
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith(".cti"):
                    found.append(os.path.join(dirpath, name))
    return found


if __name__ == "__main__":
    results = find_cti_files()
    if not results:
        print("No .cti file found under Program Files / Program Files (x86) / ProgramData.")
        print("Install a GenTL producer: Cognex GigE Vision Configuration Tool, "
              "MatrixVision mvIMPACT Acquire, or Pleora eBUS SDK.")
    else:
        print(f"{len(results)} .cti file(s) found:")
        for path in results:
            print(path)
