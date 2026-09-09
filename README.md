# quant-contest

Input files for the 2026 Huawei Algorithm Competition preliminary round
(NVFP4 -> HiF4 quantization, HiF4 conversion task).

## Contents
- `2026+Huawei+...Task+Document-0831-V1.docx` — official task document
- `how_to_solve.md` — solution guide (public score history, recommended architecture)
- `solution-0818.py` — organizer baseline solution
- `本地调试参考-0818/` — official local debugging package: `self_check.py`
  (output-format checker), baseline example, `mini_sample/linear.pt` +
  `mini_sample/attn.pt` (sample datasets), environment notes

## Note on split files
The upload network rejects HTTP bodies over ~70KB, so binary files are stored
base64-encoded in `parts/<name>/gNNN/part-NNNNN` (see `MANIFEST.json`).
Restore the originals with:

    python3 tools/reassemble.py

which verifies sha256 for every file. Text files listed above are whole.
