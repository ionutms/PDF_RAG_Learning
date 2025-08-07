import pymupdf4llm
import os
from pathlib import Path
from typing import Dict, List, Optional
import pymupdf


class PDFToMarkdownConverter:
    """
    A comprehensive PDF to Markdown converter using pymupdf4llm
    """

    def __init__(self):
        self.supported_extensions = [".pdf"]

    def convert_pdf_to_markdown(
        self,
        pdf_path: str,
        output_path: Optional[str] = None,
        page_chunks: bool = False,
        write_images: bool = False,
        image_path: Optional[str] = None,
        dpi: int = 150,
    ) -> Dict:
        """
        Convert a PDF file to markdown format

        Args:
            pdf_path: Path to the input PDF file
            output_path: Path for output markdown file (optional)
            page_chunks: If True, return content split by pages
            write_images: If True, extract and save images
            image_path: Directory to save extracted images
            dpi: DPI for image extraction

        Returns:
            Dictionary containing markdown text and metadata
        """

        # Validate input file
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.lower().endswith(".pdf"):
            raise ValueError("Input file must be a PDF")

        try:
            print(f"Converting PDF: {pdf_path}")

            # Basic conversion
            if not page_chunks:
                # Convert entire PDF to single markdown string
                markdown_text = pymupdf4llm.to_markdown(
                    pdf_path,
                    write_images=write_images,
                    image_path=image_path,
                    dpi=dpi,
                )

                result = {
                    "content": markdown_text,
                    "source_file": pdf_path,
                    "total_pages": self._get_page_count(pdf_path),
                    "conversion_type": "full_document",
                }

            else:
                # Convert with page-by-page breakdown
                markdown_pages = pymupdf4llm.to_markdown(
                    pdf_path,
                    page_chunks=True,
                    write_images=write_images,
                    image_path=image_path,
                    dpi=dpi,
                )

                result = {
                    "pages": markdown_pages,
                    "source_file": pdf_path,
                    "total_pages": len(markdown_pages),
                    "conversion_type": "page_chunks",
                }

            # Save to file if output path specified
            if output_path:
                self._save_markdown(result, output_path)
                print(f"Markdown saved to: {output_path}")

            print("Conversion completed successfully!")
            return result

        except Exception as e:
            print(f"Error during conversion: {str(e)}")
            raise

    def convert_with_advanced_options(
        self, pdf_path: str, output_path: Optional[str] = None, **kwargs
    ) -> Dict:
        """
        Convert PDF with advanced pymupdf4llm options

        Additional kwargs can include:
        - margins: (left, top, right, bottom) margins to ignore
        - dpi: Resolution for image extraction
        - page_chunks: Split by pages
        - write_images: Extract images
        - image_path: Path for extracted images
        - hdr_info: Include header information
        - show_progress: Show conversion progress
        """

        print(f"Converting with advanced options: {kwargs}")

        try:
            markdown_text = pymupdf4llm.to_markdown(pdf_path, **kwargs)

            result = {
                "content": markdown_text,
                "source_file": pdf_path,
                "conversion_options": kwargs,
                "total_pages": self._get_page_count(pdf_path),
            }

            if output_path:
                self._save_markdown(result, output_path)

            return result

        except Exception as e:
            print(f"Advanced conversion failed: {str(e)}")
            raise

    def batch_convert(
        self, input_directory: str, output_directory: str, **kwargs
    ) -> List[Dict]:
        """
        Convert multiple PDF files in a directory

        Args:
            input_directory: Directory containing PDF files
            output_directory: Directory for output markdown files
            **kwargs: Additional options for conversion

        Returns:
            List of conversion results
        """

        input_path = Path(input_directory)
        output_path = Path(output_directory)

        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all PDF files
        pdf_files = list(input_path.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {input_directory}")
            return []

        print(f"Found {len(pdf_files)} PDF files to convert")

        results = []
        for pdf_file in pdf_files:
            try:
                # Generate output filename
                output_file = output_path / f"{pdf_file.stem}.md"

                # Convert the PDF
                result = self.convert_pdf_to_markdown(
                    str(pdf_file), str(output_file), **kwargs
                )

                results.append(result)
                print(f"✓ Converted: {pdf_file.name}")

            except Exception as e:
                print(f"✗ Failed to convert {pdf_file.name}: {str(e)}")
                results.append({
                    "source_file": str(pdf_file),
                    "error": str(e),
                    "status": "failed",
                })

        return results

    def _get_page_count(self, pdf_path: str) -> int:
        """Get the number of pages in a PDF file"""

        try:
            with pymupdf.open(pdf_path) as doc:
                return doc.page_count
        except (
            FileNotFoundError,
            pymupdf.FileDataError,
            PermissionError,
        ) as e:
            print(f"Warning: Could not get page count for {pdf_path}: {e}")
            return 0
        except Exception as e:
            print(f"Unexpected error getting page count for {pdf_path}: {e}")
            return 0

    def _save_markdown(self, result: Dict, output_path: str):
        """Save markdown content to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if result.get("conversion_type") == "page_chunks":
            # Save page chunks as separate sections
            content = "# Document Content\n\n"
            for i, page in enumerate(result["pages"], 1):
                content += f"## Page {i}\n\n{page}\n\n---\n\n"
        else:
            content = result.get("content", "")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

    def preview_conversion(self, pdf_path: str, max_chars: int = 1000) -> str:
        """
        Preview the first part of the conversion without saving

        Args:
            pdf_path: Path to PDF file
            max_chars: Maximum characters to return in preview

        Returns:
            Preview of markdown content
        """

        try:
            # Convert just the first few pages for preview
            markdown_text = pymupdf4llm.to_markdown(pdf_path)

            preview = markdown_text[:max_chars]
            if len(markdown_text) > max_chars:
                preview += "\n\n[... truncated for preview ...]"

            return preview

        except Exception as e:
            return f"Preview failed: {str(e)}"


# Example usage and demonstration
def main():
    """
    Demonstration of the PDF to Markdown converter
    """
    converter = PDFToMarkdownConverter()

    # Example 1: Basic conversion
    print("=== Example 1: Basic PDF to Markdown ===")
    try:
        # Replace with your actual PDF path
        pdf_file = "./app/pdfs/max20048.pdf"

        # Basic conversion
        result = converter.convert_pdf_to_markdown(
            pdf_path=pdf_file, output_path="./app/markdowns/output_basic.md"
        )

        print(f"Converted {result['total_pages']} pages")

    except FileNotFoundError:
        print("Sample PDF not found. Please provide a valid PDF path.")
    except Exception as e:
        print(f"Conversion failed: {e}")

    # Example 2: Page-by-page conversion
    print("\n=== Example 2: Page-by-page conversion ===")
    try:
        result = converter.convert_pdf_to_markdown(
            pdf_path="./app/pdfs/max20048.pdf",
            output_path="./app/markdowns/output_pages.md",
            page_chunks=True,
        )

        print(f"Converted {result['total_pages']} pages separately")

    except Exception as e:
        print(f"Page conversion failed: {e}")

    # Example 3: Conversion with image extraction
    print("\n=== Example 3: With image extraction ===")
    try:
        result = converter.convert_pdf_to_markdown(
            pdf_path="./app/pdfs/max20048.pdf",
            output_path="./app/markdowns/output_with_images.md",
            write_images=True,
            image_path="./app/markdowns/extracted_images/",
            dpi=200,
        )

        print("Conversion with images completed")

    except Exception as e:
        print(f"Image extraction failed: {e}")

    # Example 4: Advanced options
    print("\n=== Example 4: Advanced conversion options ===")
    try:
        result = converter.convert_with_advanced_options(
            pdf_path="./app/pdfs/max20048.pdf",
            output_path="./app/markdowns/output_advanced.md",
            margins=(50, 50, 50, 50),  # Ignore margins
            dpi=150,
            show_progress=True,
        )

        print("Advanced conversion completed")

    except Exception as e:
        print(f"Advanced conversion failed: {e}")

    # Example 5: Batch conversion
    print("\n=== Example 5: Batch conversion ===")
    try:
        results = converter.batch_convert(
            input_directory="./app/pdfs/",
            output_directory="./app/markdowns/",
            write_images=True,
            image_path="./app/markdowns/batch_images/",
        )

        successful = [r for r in results if "error" not in r]
        print(
            f"Batch conversion: {len(successful)} successful, {len(results) - len(successful)} failed"
        )

    except Exception as e:
        print(f"Batch conversion failed: {e}")

    # Example 6: Preview before conversion
    print("\n=== Example 6: Preview conversion ===")
    try:
        preview = converter.preview_conversion(
            "./app/pdfs/max20048.pdf", max_chars=500
        )
        print("Preview:")
        print(preview)

    except Exception as e:
        print(f"Preview failed: {e}")


if __name__ == "__main__":
    main()
