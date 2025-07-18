# docling 의 DocumentConverter 사용

from docling.document_converter import DocumentConverter
from docling.datamodel.document import ConversionResult

def convert_pdf(pdf_path: str) -> ConversionResult:
    """
    PDF 파일을 읽어서 ConversionResult 객체로 반환함
    """
    converter = DocumentConverter()
    convert_result = converter.convert(pdf_path)
    return convert_result

def convert_pdf_to_text(pdf_path: str) -> str:
    """
    PDF 파일을 읽어서 본문 텍스트만 반환함
    """
    result = convert_pdf(pdf_path)
    # ConversionResult에서 본문 텍스트 추출
    return result.document.export_to_text()

if __name__ == "__main__":

    text = convert_pdf_to_text("data/housing_leasing_law.pdf")
    print(text[:100])