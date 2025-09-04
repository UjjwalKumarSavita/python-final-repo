from __future__ import annotations
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

def summary_to_pdf_bytes(title: str, text: str) -> bytes:
    """
    Render a simple, single-column PDF with a title and body text.
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, height - 2*cm, title)

    # Body
    c.setFont("Helvetica", 11)
    x = 2*cm
    y = height - 3*cm
    max_width = width - 4*cm
    for para in text.split("\n"):
        for line in _wrap_text(c, para, max_width):
            if y < 2*cm:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - 2*cm
            c.drawString(x, y, line)
            y -= 14
        y -= 6  # paragraph spacing

    c.showPage()
    c.save()
    return buf.getvalue()

def _wrap_text(c, text: str, max_width: float) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for w in words[1:]:
        test = current + " " + w
        if c.stringWidth(test, "Helvetica", 11) <= max_width:
            current = test
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines
