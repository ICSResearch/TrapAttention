<div align="center">

# **Trap Attention**  
### *Monocular Depth Estimation with Manual Traps — Implementation*  
*(Clean, centered layout for clarity and style.)*

</div>

---

# 📑 Table of Contents  
1. [Paper & Poster](#paper--poster)  
2. [Supplemental Materials](#supplemental-materials)  
3. [Code & Checkpoint](#code--checkpoint)  
4. [Environment](#environment)  
5. [Usage](#usage)  
6. [Citation & If This Helps, Please Cite](#citation--if-this-helps-please-cite)  
7. [Acknowledgement](#acknowledgement)  
8. [Other / TODO](#other--todo)  

---

# 📰 Paper & Poster

**Paper:**  
[Trap Attention: Monocular Depth Estimation With Manual Traps — CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ning_Trap_Attention_Monocular_Depth_Estimation_With_Manual_Traps_CVPR_2023_paper.html) :contentReference[oaicite:0]{index=0}  

**Poster & Supplement & Related Materials:**  
- Poster / Virtual Session: available via CVPR 2023 Poster Page :contentReference[oaicite:1]{index=1}  
- Supplemental PDF (network details / extra results): accessible here – [Supplemental File](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Ning_Trap_Attention_Monocular_CVPR_2023_supplemental.pdf) :contentReference[oaicite:2]{index=2}  

*(If you want, you can download the poster and place under `./docs/` or `./posters/` and insert display.)*

---

# 📦 Code & Checkpoint

- Implementation repository: [ICSResearch/TrapAttention on GitHub](https://github.com/ICSResearch/TrapAttention) :contentReference[oaicite:3]{index=3}  
- Pretrained / checkpoint models: (as you provided earlier) Google Drive link.

```text
Google Drive: https://drive.google.com/drive/folders/1kIXg9UP0cVWUq_7Pq20JT9_RyR-PjvkS?usp=sharing
```
# 🧰 Environment
Python 3.8• PyTorch 1.7.1 (or greater, as long as compatible)
(You may install other dependencies per requirements.txt in the repo.)
# ▶️ Usage / Quick Start
Clone the repo, download the checkpoint, and run as follows (for example):
```bash
git clone https://github.com/ICSResearch/TrapAttention.git
cd TrapAttention
```
# install dependencies
pip install -r requirements.txt

# example usage
```python
your_run_script.py --config configs/your_config.yaml  # adjust as needed
```
# 📝 Tip
Make sure CUDA / GPU memory is sufficient if you run high‑res inputs or large batch size.
# 📚 Citation – If This Code Helps, Please CiteIf you use this code (or parts of it) in your work, please cite:bibtex
```bash
@InProceedings{Ning_2023_CVPR,
  author    = {Chao Ning and Hongping Gan},
  title     = {Trap Attention: Monocular Depth Estimation With Manual Traps},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023},
  pages     = {5033–5043}
}
```

# 🙏 Acknowledgement
Thanks to the following outstanding works / libraries / communities:• Transformers / Vision‑Transformer backbones used in encoder• The community maintaining open‑source depth‑estimation toolboxes• 
All contributors and testers who reported bugs or improvements
# ⚠️ Other
/ TODO• (Optional) Add inference examples & sample outputs in 
/examples/• (Optional) Add visualization of depth maps (RGB → depth) in README 
/ docs• (Optional) Add evaluation scripts and result tables
