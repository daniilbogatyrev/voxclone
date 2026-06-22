from voxclone.capture.prompts import PROMPTS, get_prompts, all_prompts

def test_categories_present_and_sized():
    assert set(PROMPTS) == {"harvard", "expressive", "conversational", "technical"}
    assert len(PROMPTS["harvard"]) >= 200
    assert len(PROMPTS["expressive"]) >= 50
    assert len(PROMPTS["conversational"]) >= 20
    assert len(PROMPTS["technical"]) >= 20

def test_technical_has_numbers_or_propernouns():
    assert any(any(ch.isdigit() for ch in s) for s in PROMPTS["technical"])

def test_get_prompts_returns_list():
    assert get_prompts("harvard") == PROMPTS["harvard"]

def test_all_prompts_tags_category():
    items = all_prompts()
    assert ("expressive", PROMPTS["expressive"][0]) in items
    assert all(cat in PROMPTS for cat, _ in items)
