from voxclone.eval.wer import wer

def test_identical_is_zero():
    assert wer("the cat sat", "the cat sat") == 0.0

def test_one_substitution():
    assert abs(wer("the cat sat", "the dog sat") - 1 / 3) < 1e-6

def test_deletion_and_insertion():
    assert abs(wer("a b c d", "a c d") - 1 / 4) < 1e-6   # one deletion
    assert wer("", "anything here") > 0.0

def test_case_and_punctuation_insensitive():
    assert wer("Hello, world!", "hello world") == 0.0

def test_wer_capped_at_one():
    assert wer("hello", "hello world foo bar baz") == 1.0

def test_number_words_equal_digits():
    assert wer("I have 25 cats", "I have twenty five cats") == 0.0

def test_abbreviation_and_number_mix():
    assert wer("It costs 5 dollars", "It costs five dollars") == 0.0

def test_german_preserves_umlauts():
    # German WER must keep ä/ö/ü/ß so distinct words stay distinct. The English
    # normalizer strips diacritics ("Müller" -> "muller"), which would wrongly score
    # a real substitution as a perfect match (0.0).
    assert wer("Müller", "Muller", language="de") == 1.0

def test_german_identical_is_zero():
    assert wer("Schöne Grüße über Öl", "Schöne Grüße über Öl", language="de") == 0.0

def test_german_does_not_apply_english_abbreviation_expansion():
    # English normalizer turns "Dr." -> "doctor"; the German (basic) one must not,
    # so "Dr. Müller" vs "doctor Müller" is a real one-word error.
    assert abs(wer("Dr. Müller", "doctor Müller", language="de") - 0.5) < 1e-6
