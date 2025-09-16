import json
import os
import tempfile
from pathlib import Path

import pytest

from sqlite_rag import SQLiteRag


class TestSQLiteRag:
    def test_add_simple_text_file(self):
        #  test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "This is a test document with some content. And this is another very long sentence."
            )
            temp_file_path = f.name

        rag = SQLiteRag.create(":memory:")

        rag.add(temp_file_path)

        conn = rag._conn
        cursor = conn.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        assert doc_count == 1

        cursor = conn.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        assert chunk_count > 0

    def test_add_unsupported_file_type(self):
        # Create a temporary file with an unsupported extension
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".unsupported", delete=False
        ) as f:
            f.write("This is a test document with unsupported file type.")

            rag = SQLiteRag.create(":memory:")

            # Attempt to add the unsupported file
            processed = rag.add(f.name)

            assert processed == 0

    def test_add_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file1 = Path(temp_dir) / "file1.txt"
            file2 = Path(temp_dir) / "file2.txt"

            file1.write_text("This is the first test document.")
            file2.write_text("This is the second test document.")

            rag = SQLiteRag.create(db_path=":memory:")

            rag.add(temp_dir)

            conn = rag._conn
            cursor = conn.execute("SELECT COUNT(*) AS total FROM documents")
            doc_count = cursor.fetchone()[0]
            assert doc_count == 2

            cursor = conn.execute("SELECT COUNT(*) AS total FROM chunks")
            chunk_count = cursor.fetchone()[0]
            assert chunk_count > 0

    def test_add_directory_recursively(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sub_dir = Path(temp_dir) / "subdir"
            sub_dir.mkdir()

            file1 = Path(temp_dir) / "file1.txt"
            file2 = sub_dir / "file2.txt"

            file1.write_text("This is the first test document.")
            file2.write_text("This is the second test document in a subdirectory.")

            rag = SQLiteRag.create(":memory:")

            rag.add(temp_dir, recursive=True)

            conn = rag._conn
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            assert doc_count == 2

            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            assert chunk_count > 0

    def test_add_with_absolute_paths(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document with absolute path option.")
            temp_file_path = f.name

        rag = SQLiteRag.create(":memory:")

        rag.add(temp_file_path, use_relative_paths=False)

        conn = rag._conn
        cursor = conn.execute("SELECT uri FROM documents")
        doc = cursor.fetchone()
        assert doc
        assert doc[0] == str(Path(temp_file_path).absolute())

    def test_add_with_relative_paths(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document with relative path option.")
            temp_file_path = Path(f.name)

        rag = SQLiteRag.create(":memory:")

        rag.add(str(temp_file_path), use_relative_paths=True)

        conn = rag._conn
        cursor = conn.execute("SELECT uri FROM documents")
        doc = cursor.fetchone()
        assert doc
        assert doc[0] == str(temp_file_path.relative_to(temp_file_path.parent))

    def test_add_file_with_metadata(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document with metadata.")
            temp_file_path = f.name

        rag = SQLiteRag.create(":memory:")

        metadata = {"author": "test", "date": "2023-10-01"}

        rag.add(
            temp_file_path,
            metadata=metadata,
        )

        conn = rag._conn
        cursor = conn.execute("SELECT content, metadata FROM documents")
        doc = cursor.fetchone()
        assert doc
        assert doc[0] == "This is a test document with metadata."
        assert doc[1] == json.dumps(metadata)

    def test_add_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_file_path = f.name

        rag = SQLiteRag.create(":memory:")
        processed = rag.add(temp_file_path)

        assert processed == 0

    def test_add_unchanged_file_twice(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document that will be added twice.")
            temp_file_path = f.name

        rag = SQLiteRag.create(":memory:")

        # Add the file once
        rag.add(temp_file_path)

        conn = rag._conn
        cursor = conn.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        assert doc_count == 1

        # Add the same file again
        rag.add(temp_file_path)

        # Still should be only one document
        cursor = conn.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        assert doc_count == 1

    def test_add_text(self):
        rag = SQLiteRag.create(":memory:")

        rag.add_text(
            "This is a test document content with some text to be indexed.",
            uri="test_doc.txt",
            metadata={"author": "test"},
        )

        conn = rag._conn
        cursor = conn.execute("SELECT content, uri, metadata FROM documents")
        doc = cursor.fetchone()
        assert doc
        assert doc[0] == "This is a test document content with some text to be indexed."
        assert doc[1] == "test_doc.txt"
        assert doc[2] == '{"author": "test"}'

        cursor = conn.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        assert chunk_count > 0

    def test_add_text_without_options(self):
        rag = SQLiteRag.create(":memory:")

        rag.add_text("This is a test document content without options.")

        conn = rag._conn
        cursor = conn.execute("SELECT content FROM documents")
        doc = cursor.fetchone()
        assert doc
        assert doc[0] == "This is a test document content without options."

        cursor = conn.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        assert chunk_count > 0

    def test_add_text_with_metadata(self):
        rag = SQLiteRag.create(":memory:")

        metadata = {"author": "test", "date": "2023-10-01"}

        rag.add_text(
            "This is a test document content with metadata.",
            uri="test_doc_with_metadata.txt",
            metadata=metadata,
        )

        conn = rag._conn
        cursor = conn.execute("SELECT content, uri, metadata FROM documents")
        doc = cursor.fetchone()
        assert doc
        assert doc[0] == "This is a test document content with metadata."
        assert doc[1] == "test_doc_with_metadata.txt"
        assert doc[2] == json.dumps(metadata)

    def test_list_documents(self):
        rag = SQLiteRag.create(":memory:")

        rag.add_text("Document 1 content.")
        rag.add_text("Document 2 content.")

        documents = rag.list_documents()
        assert len(documents) == 2
        assert documents[0].content == "Document 1 content."
        assert documents[1].content == "Document 2 content."

    def test_find_document_by_id(self):
        rag = SQLiteRag.create(":memory:")

        rag.add_text(
            "Test document content.", uri="test.txt", metadata={"author": "test"}
        )
        documents = rag.list_documents()
        doc_id = documents[0].id

        # Find by ID
        assert doc_id is not None
        found_doc = rag.find_document(doc_id)

        assert found_doc is not None
        assert found_doc.id == doc_id
        assert found_doc.content == "Test document content."
        assert found_doc.uri == "test.txt"
        assert found_doc.metadata == {"author": "test"}

    def test_find_document_by_uri(self):
        rag = SQLiteRag.create(":memory:")

        rag.add_text(
            "Test document content.", uri="test.txt", metadata={"author": "test"}
        )

        # Find by URI
        found_doc = rag.find_document("test.txt")

        assert found_doc is not None
        assert found_doc.content == "Test document content."
        assert found_doc.uri == "test.txt"
        assert found_doc.metadata == {"author": "test"}

    def test_find_document_not_found(self):
        rag = SQLiteRag.create(":memory:")

        found_doc = rag.find_document("nonexistent")

        assert found_doc is None

    def test_remove_document_by_id(self):
        rag = SQLiteRag.create(":memory:")

        rag.add_text(
            "Test document content.", uri="test.txt", metadata={"author": "test"}
        )
        documents = rag.list_documents()
        doc_id = documents[0].id

        # Verify document exists
        assert len(documents) == 1

        # Remove by ID
        assert doc_id is not None
        success = rag.remove_document(doc_id)

        assert success is True

        # Verify document is removed
        documents = rag.list_documents()
        assert len(documents) == 0

    def test_remove_document_by_uri(self):
        rag = SQLiteRag.create(":memory:")

        rag.add_text(
            "Test document content.", uri="test.txt", metadata={"author": "test"}
        )

        # Verify document exists
        documents = rag.list_documents()
        assert len(documents) == 1

        # Remove by URI
        success = rag.remove_document("test.txt")

        assert success is True

        # Verify document is removed
        documents = rag.list_documents()
        assert len(documents) == 0

    def test_remove_document_not_found(self):
        rag = SQLiteRag.create(":memory:")

        success = rag.remove_document("nonexistent")

        assert success is False

    def test_remove_document_with_chunks(self):
        rag = SQLiteRag.create(":memory:")

        # Add document that will create chunks
        rag.add_text(
            "This is a longer document that should create multiple chunks when processed by the chunker.",
            uri="test.txt",
        )

        # Verify document and chunks exist
        documents = rag.list_documents()
        assert len(documents) == 1
        doc_id = documents[0].id

        cursor = rag._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,))
        chunk_count = cursor.fetchone()[0]
        assert chunk_count > 0

        # Remove document
        assert doc_id is not None
        success = rag.remove_document(doc_id)

        assert success is True

        # Verify document and chunks are removed
        documents = rag.list_documents()
        assert len(documents) == 0

        cursor.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,))
        chunk_count = cursor.fetchone()[0]
        assert chunk_count == 0

    def test_rebuild_with_existing_files(self):
        """Test rebuild with files that still exist"""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
            f1.write("Original content for file 1")
            file1_path = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
            f2.write("Original content for file 2")
            file2_path = f2.name

        rag = SQLiteRag.create(":memory:")

        rag.add(file1_path)
        rag.add(file2_path)

        documents = rag.list_documents()
        assert len(documents) == 2

        cursor = rag._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        initial_chunk_count = cursor.fetchone()[0]
        assert initial_chunk_count > 0

        # Act
        Path(file1_path).write_text("Modified content for file 1")
        Path(file2_path).write_text("Modified content for file 2")

        result = rag.rebuild()

        # Assert
        assert result["total"] == 2
        assert result["reprocessed"] == 2
        assert result["not_found"] == 0
        assert result["removed"] == 0

        documents = rag.list_documents()
        assert len(documents) == 2

        # Check that content was updated
        found_file1 = None
        found_file2 = None
        for doc in documents:
            if doc.uri == file1_path:
                found_file1 = doc
            elif doc.uri == file2_path:
                found_file2 = doc

        assert found_file1 is not None
        assert found_file2 is not None
        assert "Modified content for file 1" in found_file1.content
        assert "Modified content for file 2" in found_file2.content

    def test_rebuild_with_missing_files(self):
        """Test rebuild when some files are missing"""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
            f1.write("Content for file 1")
            file1_path = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
            f2.write("Content for file 2")
            file2_path = f2.name

        rag = SQLiteRag.create(":memory:")

        # Add files
        rag.add(file1_path)
        rag.add(file2_path)

        # Make file missing
        os.unlink(file2_path)

        # Act
        result = rag.rebuild(remove_missing=False)

        # Assert
        assert result["total"] == 2
        assert result["reprocessed"] == 1  # Only file1 was reprocessed
        assert result["not_found"] == 1  # file2 was not found
        assert result["removed"] == 0  # Nothing removed

        # Both documents should still exist in database
        documents = rag.list_documents()
        assert len(documents) == 2

    def test_rebuild_remove_missing_files(self):
        """Test rebuild with remove_missing=True"""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
            f1.write("Content for file 1")
            file1_path = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
            f2.write("Content for file 2")
            file2_path = f2.name

        rag = SQLiteRag.create(":memory:")

        rag.add(file1_path)
        rag.add(file2_path)

        # Make it missing
        os.unlink(file2_path)

        # Act
        result = rag.rebuild(remove_missing=True)

        # Assert
        assert result["total"] == 2
        assert result["reprocessed"] == 1  # Only file1 was reprocessed
        assert result["not_found"] == 1  # file2 was not found
        assert result["removed"] == 1  # file2 was removed

        # Only one document should remain
        documents = rag.list_documents()
        assert len(documents) == 1
        assert documents[0].uri == file1_path

    def test_rebuild_text_documents(self):
        """Test rebuild with text documents (no URI)"""
        rag = SQLiteRag.create(":memory:")

        rag.add_text("Text document 1 content")

        result = rag.rebuild()

        assert result["total"] == 1
        assert result["reprocessed"] == 1
        assert result["not_found"] == 0
        assert result["removed"] == 0

        documents = rag.list_documents()
        assert len(documents) == 1

    def test_reset_database(self):
        temp_file_path = os.path.join(tempfile.mkdtemp(), "something")

        rag = SQLiteRag.create(temp_file_path)

        rag.add_text("Test document 1")
        rag.add_text("Test document 2", uri="test.txt")

        documents = rag.list_documents()
        assert len(documents) == 2

        success = rag.reset()
        assert success is True

        assert not Path(temp_file_path).exists()

    def test_search_exact_match(self):
        # cosin distance for searching embedding is exact 0.0 when strings match
        settings = {"other_vector_options": "distance=cosine"}

        temp_file_path = os.path.join(tempfile.mkdtemp(), "something")

        rag = SQLiteRag.create(temp_file_path, settings=settings)

        expected_string = "The quick brown fox jumps over the lazy dog"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
            f1.write(expected_string)
            file1_path = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
            f2.write(
                "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
            )
            file2_path = f2.name

        rag.add(file1_path)
        rag.add(file2_path)

        # Act
        results = rag.search(expected_string)

        assert len(results) > 0
        assert expected_string == results[0].document.content
        assert 1 == results[0].vec_rank
        assert 0.0 == results[0].vec_distance

    @pytest.mark.parametrize(
        "quantize_scan", [True, False], ids=["quantize", "no-quantize"]
    )
    def test_search_samples_exact_match_by_scan_type(self, quantize_scan: bool):
        # Test that searching for exact content from sample files returns distance 0
        # FTS not included in the combined score
        settings = {
            "other_vector_options": "distance=cosine",
            "weight_fts": 0.0,
            "quantize_scan": quantize_scan,
        }

        temp_file_path = os.path.join(tempfile.mkdtemp(), "mydb.db")
        rag = SQLiteRag.create(temp_file_path, settings=settings)

        # Index all sample files
        samples_dir = Path(__file__).parent / "assets" / "samples"
        rag.add(str(samples_dir))

        # Get all sample files and test each one
        sample_files = list(samples_dir.glob("*.txt"))

        for sample_file in sample_files:
            file_content = sample_file.read_text(encoding="utf-8")

            # Search for the exact content
            results = rag.search(file_content, top_k=2)

            assert len(results) == 2

            first_result = results[0]
            assert first_result.vec_distance == 0.0
            assert first_result.document.content == file_content

            # Second result should have distance > 0
            second_result = results[1]
            assert second_result.vec_distance and second_result.vec_distance > 0.0
