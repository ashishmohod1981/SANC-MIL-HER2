import sys, nbformat, re
from pathlib import Path
src = Path(__file__).parent / "src"
parts = sorted(src.glob("p*.py"))
code = "\n\n".join(p.read_text() for p in parts)
Path(__file__).parent.joinpath("her2_wsl_pipeline.py").write_text(code)
cells = []
for chunk in re.split(r"^# %%", code, flags=re.M):
    if not chunk.strip(): continue
    if chunk.startswith(" [markdown]"):
        body = chunk[len(" [markdown]"):].strip("\n")
        md = "\n".join(l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l) for l in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(md))
    else:
        cells.append(nbformat.v4.new_code_cell(chunk.strip("\n")))
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
     "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}})
nbformat.write(nb, Path(__file__).parent / "her2_wsl_kaggle.ipynb")
print(len(cells), "cells")
