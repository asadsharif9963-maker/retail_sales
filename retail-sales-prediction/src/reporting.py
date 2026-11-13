import io
import pandas as pd


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(sheets):
    """sheets: {sheet_name: DataFrame}"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            safe = name[:31]
            df.to_excel(xw, sheet_name=safe, index=False)
    return buf.getvalue()


def to_pdf_bytes(title, summary_lines, df=None):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=styles["Heading1"], fontSize=18, spaceAfter=12)
    body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=10, leading=15)

    story = [Paragraph(title, h), Spacer(1, 8)]
    for line in summary_lines:
        story.append(Paragraph(line, body))
    story.append(Spacer(1, 14))

    if df is not None and len(df):
        head = df.head(30)
        data = [list(head.columns)] + head.astype(str).values.tolist()
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2A6F62")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTSIZE",   (0,0), (-1,-1), 8),
            ("GRID",       (0,0), (-1,-1), 0.25, colors.HexColor("#D9D6CE")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7F6F2")]),
        ]))
        story.append(t)
    doc.build(story)
    return buf.getvalue()
