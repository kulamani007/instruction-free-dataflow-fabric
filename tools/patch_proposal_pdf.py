"""One-off (applied 1 Nov 2026, commit 1a2d81e): patch the proposal PDF without rebuilding it.

Replaces the contact email on page 1 and inserts the "FPGA overlays (DLA [13])" bullet in
section 6 on page 3, shifting the rest of page 3 down, using the PDF's own embedded subset
fonts (a ToUnicode-derived cmap is rebuilt so new text can be set). Kept for reference:
the original pandoc + wkhtmltopdf build and Figure 1 source were not available on the Mac.
Re-running it on the current PDF fails by design (the old email is gone).

Usage: python tools/patch_proposal_pdf.py in.pdf out.pdf      (needs pymupdf, fonttools)
"""
import io, re, sys
import fitz
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._c_m_a_p import cmap_format_4

src_path, out_path = sys.argv[1], sys.argv[2]
src = fitz.open(src_path)


def font_with_cmap(doc, page, basename):
    """Return fitz.Font built from the embedded subset, with a cmap rebuilt from ToUnicode."""
    xref = next(f[0] for f in page.get_fonts() if f[3] == basename)
    desc = int(re.search(r"(\d+) 0 R", doc.xref_get_key(xref, "DescendantFonts")[1]).group(1))
    assert doc.xref_get_key(desc, "CIDToGIDMap")[1] in ("/Identity", "null"), "non-identity CID map"
    tu = int(doc.xref_get_key(xref, "ToUnicode")[1].split()[0])
    cm = doc.xref_stream(tu).decode("latin-1")
    uni = {}
    for block in re.findall(r"beginbfchar(.*?)endbfchar", cm, re.S):
        for cid, u in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            if len(u) == 4:
                uni[int(u, 16)] = int(cid, 16)
    for block in re.findall(r"beginbfrange(.*?)endbfrange", cm, re.S):
        for a, b, u in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            for k in range(int(a, 16), int(b, 16) + 1):
                uni[int(u, 16) + k - int(a, 16)] = k
        for a, arr in re.findall(r"<([0-9A-Fa-f]+)>\s*<[0-9A-Fa-f]+>\s*\[(.*?)\]", block, re.S):
            for k, u in enumerate(re.findall(r"<([0-9A-Fa-f]+)>", arr)):
                uni[int(u, 16)] = int(a, 16) + k
    if 0x20 not in uni and 0x09 in uni:          # Qt subsets map the space glyph to U+0009
        uni[0x20] = uni[0x09]
    buf = doc.extract_font(xref)[3]
    tt = TTFont(io.BytesIO(buf))
    order = tt.getGlyphOrder()
    sub = cmap_format_4(4)
    sub.platformID, sub.platEncID, sub.language = 3, 1, 0
    sub.cmap = {u: order[g] for u, g in uni.items() if g < len(order) and u < 0x10000}
    tt["cmap"] = newTable("cmap")
    tt["cmap"].tableVersion = 0
    tt["cmap"].tables = [sub]
    out = io.BytesIO()
    tt.save(out)
    return fitz.Font(fontbuffer=out.getvalue())


def check(font, text):
    missing = sorted({c for c in text if not font.has_glyph(ord(c))})
    assert not missing, f"{font.name}: glyphs missing for {missing}"


SIZE = 8.65

# ------------------------------------------------------------------ page 1: email
p1 = src[0]
reg1 = font_with_cmap(src, p1, "TeXGyreTermes-Regular")
span = next(s for b in p1.get_text("rawdict")["blocks"] for l in b.get("lines", [])
            for s in l["spans"] if "routsarojini901" in "".join(c["c"] for c in s["chars"]))
chars = span["chars"]
i = "".join(c["c"] for c in chars).index("routsarojini901")
x0, base = chars[i]["origin"]
color = fitz.sRGB_to_pdf(span["color"])
tail = "routkulamani50@gmail.com · github.com/kulamani007"
check(reg1, tail)
p1.add_redact_annot(fitz.Rect(x0 - 0.2, span["bbox"][1], span["bbox"][2] + 1, span["bbox"][3]), fill=False)
p1.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE)
tw = fitz.TextWriter(p1.rect, color=color)
tw.append((x0, base), tail, font=reg1, fontsize=SIZE)
tw.write_text(p1)

# ------------------------------------------------------------------ page 3: DLA bullet
P3 = 2
p3 = src[P3]
reg3 = font_with_cmap(src, p3, "TeXGyreTermes-Regular")
bold3 = font_with_cmap(src, p3, "TeXGyreTermes-Bold")
lines = [l for b in p3.get_text("dict")["blocks"] for l in b.get("lines", [])]
first = lambda prefix: next(l for l in lines if l["spans"][0]["text"].startswith(prefix))
stream_l = first("Streaming FPGA generators")
rt_l = first("Real-time DNN scheduling")
pitch = rt_l["spans"][0]["origin"][1] - stream_l["spans"][0]["origin"][1]    # one 2-line bullet
lead = 11.53                                                                  # line advance
split_y = rt_l["bbox"][1] - 0.5                                            # just above "Real-time"
left, right = stream_l["bbox"][0], max(l["bbox"][2] for l in lines)
color3 = fitz.sRGB_to_pdf(stream_l["spans"][0]["color"])

# bullet dot of the Streaming bullet, to copy
dot = next(d for d in p3.get_drawings()
           if d["rect"].width < 4 and stream_l["bbox"][1] < d["rect"].y0 < stream_l["bbox"][3])

# bottom part of the page, content truly removed from the top (no hidden duplicates)
tmp = fitz.open()
tmp.insert_pdf(src, from_page=P3, to_page=P3)
tmp[0].add_redact_annot(fitz.Rect(0, 0, p3.rect.width, split_y), fill=False)
tmp[0].apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                        graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED)
# remove the bottom part from the real page, then re-place it shifted down by one bullet
p3.add_redact_annot(fitz.Rect(0, split_y, p3.rect.width, p3.rect.height), fill=False)
p3.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                    graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED)
p3.show_pdf_page(p3.rect + (0, pitch, 0, pitch), tmp, 0)

# new bullet, justified like the surrounding text
base = rt_l["spans"][0]["origin"][1]
r = dot["rect"] + (0, pitch, 0, pitch)
p3.draw_oval(r, color=dot.get("color") or dot["fill"], fill=dot["fill"], width=dot.get("width") or 0)

words = [(w, bold3) for w in "FPGA overlays".split()] + \
        [(w, reg3) for w in ("(DLA [13]) run different models on one bitstream, but an instruction stream "
                             "generated by the compiler still dispatches each layer at run time, so "
                             "instruction control remains in the inference loop.").split()]
for w, f in words:
    check(f, w)
space = reg3.text_length(" ", SIZE)
rows, cur = [], []
for w in words:
    trial = cur + [w]
    width = sum(f.text_length(t, SIZE) for t, f in trial) + space * (len(trial) - 1)
    if cur and left + width > right:
        rows.append(cur); cur = [w]
    else:
        cur = trial
rows.append(cur)
assert len(rows) == 2, f"bullet wraps to {len(rows)} lines; shift assumes 2"

tw = fitz.TextWriter(p3.rect, color=color3)
for k, row in enumerate(rows):
    y = base + k * lead
    widths = [f.text_length(t, SIZE) for t, f in row]
    gap = space if k == len(rows) - 1 else (right - left - sum(widths)) / (len(row) - 1)
    x = left
    for (t, f), wdt in zip(row, widths):
        tw.append((x, y), t, font=f, fontsize=SIZE)
        x += wdt + gap
tw.write_text(p3)

assert max(b[3] for b in p3.get_text("blocks")) < p3.rect.height - 40, "content pushed into bottom margin"
src.save(out_path, garbage=4, deflate=True)
print("rows:", [" ".join(t for t, _ in r) for r in rows])
