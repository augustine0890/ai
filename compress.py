"""PDF compression using PyMuPDF (fitz) for high-quality rasterization and compression."""
import os

import fitz  # PyMuPDF

input_file = "data/wang.pdf"
output_file = "data/compressed_output.pdf"
target_size_kb = 600

# Get original size
original_size = os.path.getsize(input_file)
print(f"Original size: {original_size:,} bytes ({original_size / 1024:.1f} KB)")

# Open PDF
doc = fitz.open(input_file)
print(f"Pages: {len(doc)}")

# Create new PDF with compressed images
new_doc = fitz.open()

for page_num, page in enumerate(doc, 1):
    # Get page dimensions
    rect = page.rect
    
    # Calculate DPI for target size (lower DPI = smaller file, but still readable)
    # 100 DPI is good balance for documents
    dpi = 100
    
    # Render page to pixmap (rasterize)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # Create new page with same dimensions
    new_page = new_doc.new_page(width=rect.width, height=rect.height)
    
    # Insert the rasterized image
    new_page.insert_image(rect, pixmap=pix)
    
    print(f"  Page {page_num}: {int(rect.width)}x{int(rect.height)} @ {dpi} DPI")

# Save with compression
new_doc.save(
    output_file,
    garbage=4,  # Maximum garbage collection
    deflate=True,  # Compress streams
    deflate_images=True,  # Compress images
    deflate_fonts=True,  # Compress fonts
    clean=True,  # Clean up redundancies
)

doc.close()
new_doc.close()

# Check result
new_size = os.path.getsize(output_file)
reduction = (1 - new_size / original_size) * 100
print(f"\nNew size: {new_size:,} bytes ({new_size / 1024:.1f} KB)")
print(f"Reduction: {reduction:.1f}%")
