#!/usr/bin/env python3
"""Convert NLP reference sheet markdown to PDF with formatting"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path

# Read the markdown file
md_file = Path("nlp_reference_sheet.md")
md_content = md_file.read_text()

# Convert markdown to HTML with extensions
html_content = markdown.markdown(
    md_content,
    extensions=['tables', 'fenced_code', 'codehilite']
)

# Create a styled HTML document with compact formatting for reference sheet
styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        @page {{
            size: A4;
            margin: 0.5cm;
            margin-bottom: 1cm;
            @bottom-right {{
                content: "Page " counter(page) " of " counter(pages);
                font-size: 8pt;
                color: #666;
            }}
        }}
        body {{
            font-family: 'Arial', 'Helvetica', sans-serif;
            font-size: 8pt;
            line-height: 1.2;
            margin: 0;
            padding: 0.3cm;
            color: #000;
        }}
        h1 {{
            font-size: 16pt;
            margin: 0.2cm 0;
            padding: 0.1cm;
            background: #2c3e50;
            color: white;
            text-align: center;
            page-break-after: avoid;
        }}
        h2 {{
            font-size: 12pt;
            margin: 0.15cm 0 0.1cm 0;
            padding: 0.05cm 0.1cm;
            background: #34495e;
            color: white;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 10pt;
            margin: 0.1cm 0 0.05cm 0;
            padding: 0.05cm 0.1cm;
            background: #ecf0f1;
            color: #2c3e50;
            border-left: 3px solid #3498db;
            page-break-after: avoid;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 7pt;
            margin: 0.1cm 0;
            page-break-inside: avoid;
        }}
        th {{
            background: #3498db;
            color: white;
            padding: 0.05cm 0.1cm;
            text-align: left;
            font-weight: bold;
            border: 1px solid #2980b9;
        }}
        td {{
            padding: 0.05cm 0.1cm;
            border: 1px solid #bdc3c7;
            vertical-align: top;
        }}
        tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        code {{
            font-family: 'Courier New', monospace;
            font-size: 7pt;
            background: #f4f4f4;
            padding: 0.02cm 0.05cm;
            border-radius: 2px;
        }}
        pre {{
            font-family: 'Courier New', monospace;
            font-size: 6.5pt;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-left: 3px solid #3498db;
            padding: 0.1cm;
            margin: 0.1cm 0;
            overflow-x: auto;
            line-height: 1.1;
            page-break-inside: avoid;
        }}
        pre code {{
            background: none;
            padding: 0;
        }}
        hr {{
            border: none;
            border-top: 2px solid #3498db;
            margin: 0.2cm 0;
        }}
        p {{
            margin: 0.05cm 0;
        }}
        ul, ol {{
            margin: 0.05cm 0;
            padding-left: 0.4cm;
        }}
        li {{
            margin: 0.02cm 0;
        }}
        strong {{
            color: #2c3e50;
        }}
        /* Page break control */
        .page-break {{
            page-break-before: always;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

# Convert HTML to PDF
print("Generating PDF from markdown...")
HTML(string=styled_html).write_pdf(
    "nlp_reference_sheet.pdf",
    stylesheets=[CSS(string="@page { size: A4; margin: 0.5cm; }")]
)

print("✓ PDF created successfully: nlp_reference_sheet.pdf")
print(
    f"✓ File size: {Path('nlp_reference_sheet.pdf').stat().st_size / 1024:.1f} KB")
