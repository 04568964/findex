from findex.tokenizer import tokenizer


def test_basic_words():
    text = "Hello World"
    assert list(tokenizer(text)) == ["hello", "world"]

def test_punctuation_removal():
    text = "Hello, world! How are you?"
    assert list(tokenizer(text)) == ["hello", "world", "how", "are", "you"]

def test_hyphens_and_apostrophes():
    text = "Don't use out-of-date tools"
    assert list(tokenizer(text)) == ["don't", "use", "out-of-date", "tools"]

def test_unicode_normalization():
    text = "It’s a test"
    assert list(tokenizer(text)) == ["it's", "a", "test"]

def test_numbers_and_trailing_punctuation():
    text = "In 2024, Python is great-"
    assert list(tokenizer(text)) == ["in", "2024", "python", "is", "great"]