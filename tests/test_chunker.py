from src.utils.chunker import clean, estimate_tokens, split_sentences, chunk_document

def test_clean():
    assert clean("Hello   world!\n\n\nHow are you?") == "Hello world!\n\nHow are you?"

def test_estimate_tokens():
    assert estimate_tokens("1234") == 1
    assert estimate_tokens("12345678") == 2

def test_split_sentences():
    assert split_sentences("Hello! How are you? I am fine.") == ['Hello!', 'How are you?', 'I am fine.']

def test_chunk_document():
    doc = {"id": "1", "text": "This is a short test document. It has exactly two sentences."}
    chunks = chunk_document(doc, max_tokens=10, min_tokens=0)
    assert len(chunks) > 0
