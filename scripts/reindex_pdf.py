import sys
from pathlib import Path

# Ensure repository root is importable when executed as a file path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.session import SessionLocal, init_database
from tools.vector_store import ingest_pdf_to_vector_store


def main() -> int:
    init_database()
    db = SessionLocal()
    try:
        pdf = Path('./data/medical_book.pdf')
        if not pdf.exists():
            print('PDF not found at ./data/medical_book.pdf')
            return 1
        else:
            report = ingest_pdf_to_vector_store(db, str(pdf))
            print(f'Indexed document: {report.doc_id}')
            print(f'Parsed elements: {report.parsed_elements}')
            print(f'Chunks: {report.chunks}')
            print(f'Embeddings: {report.embeddings}')
            print(f'Inserted chunks: {report.inserted}')
            print(f'Total chunks in store: {report.total_chunks}')
            return 0
    finally:
        db.close()


if __name__ == '__main__':
    raise SystemExit(main())
