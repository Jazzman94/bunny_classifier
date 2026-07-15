from bunny_classifier.labels import LABEL_TO_INDEX, LABELS


def test_labels_match_roadmap_contract() -> None:
    assert LABELS == ("back", "cleaning", "lying", "moving", "side", "sitting", "standing")


def test_labels_are_sorted_and_unique() -> None:
    assert list(LABELS) == sorted(set(LABELS))


def test_label_to_index_roundtrip() -> None:
    for i, label in enumerate(LABELS):
        assert LABEL_TO_INDEX[label] == i
