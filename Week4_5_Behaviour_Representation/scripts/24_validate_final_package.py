from pathlib import Path
import zipfile
import shutil


ROOT = Path("Week4_5_Behaviour_Representation")
FINAL = ROOT / "final_outputs"
PKG = FINAL / "Week4_5_Final_Package"
ZIP_PATH = FINAL / "Week4_5_Behaviour_Representation_Final_Package.zip"
VALIDATION = FINAL / "Week4_5_Final_Package_Validation.md"

if not PKG.exists():
    raise FileNotFoundError(PKG)

if not ZIP_PATH.exists():
    raise FileNotFoundError(ZIP_PATH)

# Rebuild manifest after all files exist.
manifest_path = PKG / "PACKAGE_MANIFEST.tsv"
files = sorted([p for p in PKG.rglob("*") if p.is_file() and p.name != "PACKAGE_MANIFEST.tsv"])

with open(manifest_path, "w") as f:
    f.write("size_bytes\trelative_path\n")
    for p in files:
        f.write(f"{p.stat().st_size}\t{p.relative_to(PKG)}\n")

# Add manifest itself at the end with final size.
current_manifest_size = manifest_path.stat().st_size
with open(manifest_path, "a") as f:
    f.write(f"{current_manifest_size}\tPACKAGE_MANIFEST.tsv\n")

# Rebuild ZIP after manifest fix.
ZIP_PATH.unlink()

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(FINAL))

# Validate ZIP.
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    bad_file = z.testzip()
    zip_entries = len(z.infolist())

package_files = len([p for p in PKG.rglob("*") if p.is_file()])
package_size_mb = sum(p.stat().st_size for p in PKG.rglob("*") if p.is_file()) / (1024 * 1024)
zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)

with open(VALIDATION, "w") as f:
    f.write("# Week 4-5 Final Package Validation\n\n")
    f.write(f"- Package folder: `{PKG}`\n")
    f.write(f"- ZIP file: `{ZIP_PATH}`\n")
    f.write(f"- Package file count: {package_files}\n")
    f.write(f"- ZIP entry count: {zip_entries}\n")
    f.write(f"- Package size MB: {package_size_mb:.2f}\n")
    f.write(f"- ZIP size MB: {zip_size_mb:.2f}\n")
    f.write(f"- ZIP integrity result: {'OK' if bad_file is None else 'FAILED: ' + bad_file}\n\n")

    if bad_file is None and package_files == zip_entries:
        f.write("Decision: Final package validation passed.\n")
    else:
        f.write("Decision: Final package needs review.\n")

print("Saved:", VALIDATION)
print("Package files:", package_files)
print("ZIP entries:", zip_entries)
print("Package size MB:", round(package_size_mb, 2))
print("ZIP size MB:", round(zip_size_mb, 2))
print("ZIP integrity:", "OK" if bad_file is None else bad_file)
