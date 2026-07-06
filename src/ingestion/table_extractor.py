import pdfplumber

def extract_tables_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts tables from a PDF using pdfplumber and formats them as Markdown.
    Returns a list of dicts: {"page_number": int, "markdown_table": str}
    """
    tables_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            extracted_tables = page.extract_tables()
            
            for table in extracted_tables:
                if not table:
                    continue
                    
                # Clean the table data (remove newlines from cells)
                cleaned_table = []
                for row in table:
                    cleaned_row = [" ".join(str(cell).replace('\n', ' ').split()) if cell else "" for cell in row]
                    cleaned_table.append(cleaned_row)
                    
                # Format as Markdown
                if not cleaned_table or not cleaned_table[0]:
                    continue
                    
                num_cols = len(cleaned_table[0])
                
                # Header row
                markdown_str = "|" + "|".join(cleaned_table[0]) + "|\n"
                
                # Separator row
                markdown_str += "|" + "|".join(["---"] * num_cols) + "|\n"
                
                # Data rows
                for row in cleaned_table[1:]:
                    markdown_str += "|" + "|".join(row) + "|\n"
                    
                tables_data.append({
                    "page_number": page_num,
                    "markdown_table": markdown_str
                })
                
    return tables_data
