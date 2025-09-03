from backend.services.entity_extraction import extract_entities

def test_entity_extraction_basic():
    text = "Alice met Bob on 2023-05-01 in Paris."
    ents = extract_entities(text)
    assert "Alice" in ents.get("names", []) or "Bob" in ents.get("names", [])
