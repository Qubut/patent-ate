"""Coordinated modifier pieces and fragment admission on the host parse."""

from __future__ import annotations

from functools import cache

import pytest
from jate.models import Document
from spacy.language import Language
from spacy.tokens import Doc

from patent_ate.nlp import JateDraw, host_language
from patent_ate.spec import AteSpec


@cache
def host() -> Language:
    """Return the packaged spaCy model used by the extractor."""
    return host_language(AteSpec().spacy_model)


def pieces_of(sentence: str, phrase: str) -> tuple[tuple[str, str, str], ...]:
    """Return (key, surface, covered text) for the pieces of ``phrase`` inside ``sentence``."""
    doc: Doc = host()(sentence)
    start = sentence.index(phrase)
    end = start + len(phrase)
    return tuple(
        (key, surface, doc.text[s:e])
        for key, surface, s, e in JateDraw.pieces(doc, start, end, phrase.lower(), phrase)
    )


def test_two_conjunct_modifiers_expand_to_one_piece_each() -> None:
    result = pieces_of(
        'The cell uses a lithium-ion or solid-state battery in a vehicle.',
        'a lithium-ion or solid-state battery',
    )
    keys = {key for key, _surface, _text in result}
    assert keys == {'lithium-ion battery', 'solid-state battery'}
    covered = {key: text for key, _surface, text in result}
    assert covered['lithium-ion battery'] == 'lithium-ion'
    assert covered['solid-state battery'] == 'solid-state battery'


def test_piece_offsets_index_the_source_text() -> None:
    sentence = 'The cell uses a lithium-ion or solid-state battery in a vehicle.'
    phrase = 'a lithium-ion or solid-state battery'
    doc: Doc = host()(sentence)
    start = sentence.index(phrase)
    for _key, surface, s, e in JateDraw.pieces(doc, start, start + len(phrase), phrase, phrase):
        assert 0 <= s < e <= len(sentence)
        assert surface.startswith(sentence[s:e])
        assert surface.endswith('battery')


def test_three_conjuncts_yield_three_pieces() -> None:
    result = pieces_of(
        'The frame has a copper, silver or gold layer on a substrate.',
        'a copper, silver or gold layer',
    )
    keys = {key for key, _surface, _text in result}
    assert keys == {'copper layer', 'silver layer', 'gold layer'}


def test_fixed_term_expands_and_the_spurious_key_is_a_separate_key() -> None:
    result = pieces_of('The strap uses hook and loop fasteners.', 'hook and loop fasteners')
    keys = {key for key, _surface, _text in result}
    assert keys == {'hook fastener', 'loop fastener'}


def test_numeric_coordination_keeps_the_whole_span() -> None:
    sentence = 'The jaws have the first and second jaws.'
    phrase = 'the first and second jaws'
    doc: Doc = host()(sentence)
    start = sentence.index(phrase)
    assert JateDraw.pieces(doc, start, start + len(phrase), phrase, phrase) == (
        (phrase, phrase, start, start + len(phrase)),
    )


def test_single_conjunct_keeps_the_whole_span() -> None:
    sentence = 'The device has a lithium battery.'
    phrase = 'a lithium battery'
    assert pieces_of(sentence, phrase) == ((phrase, phrase, phrase),)


@pytest.mark.parametrize(
    ('sentence', 'phrase'),
    [
        ('The method uses at least one notification.', 'at least one notification'),
        ('The message carries seq id no:405 in its header.', 'seq id no:405'),
        ('The system receives one or more sensors.', 'one or more sensors'),
    ],
)
def test_fragments_are_not_admitted(sentence: str, phrase: str) -> None:
    doc: Doc = host()(sentence)
    start = sentence.index(phrase)
    assert JateDraw.pieces(doc, start, start + len(phrase), phrase.lower(), phrase) == ()


def test_real_terms_pass_the_fragment_rule() -> None:
    sentence = 'The cell stores charge in a lithium-ion battery.'
    phrase = 'a lithium-ion battery'
    assert pieces_of(sentence, phrase) == ((phrase, phrase, phrase),)


@pytest.mark.parametrize(
    ('sentence', 'phrase'),
    [
        ('The rig holds a plurality of sensors in a frame.', 'a plurality of sensors'),
        ('The rig holds a number of sensors in a frame.', 'a number of sensors'),
        ('The rig holds a set of sensors in a frame.', 'a set of sensors'),
    ],
)
def test_quantity_noun_phrases_are_not_admitted(sentence: str, phrase: str) -> None:
    doc: Doc = host()(sentence)
    start = sentence.index(phrase)
    assert JateDraw.pieces(doc, start, start + len(phrase), phrase.lower(), phrase) == ()


def test_quantifier_outside_the_key_keeps_the_key() -> None:
    sentence = 'The method is shown in one or more non-limiting embodiments.'
    phrase = 'non-limiting embodiments'
    doc: Doc = host()(sentence)
    start = sentence.index(phrase)
    assert JateDraw.pieces(doc, start, start + len(phrase), phrase, phrase) == (
        (phrase, phrase, start, start + len(phrase)),
    )


def test_figure_labels_are_not_split() -> None:
    sentence = 'The valve has a 2a or 2b arrow in a frame.'
    phrase = 'a 2a or 2b arrow'
    doc: Doc = host()(sentence)
    start = sentence.index(phrase)
    assert JateDraw.pieces(doc, start, start + len(phrase), phrase.lower(), phrase) == (
        (phrase.lower(), phrase, start, start + len(phrase)),
    )


def extracted(*texts: str) -> dict[str, list[str]]:
    """Return each extracted key with its sorted surface forms."""
    documents = tuple(Document(doc_id=str(index), content=text) for index, text in enumerate(texts))
    parsed = tuple(host().pipe(texts))
    return {
        cand.normalized_form: sorted(cand.surface_forms)
        for cand in JateDraw.extract(documents, parsed)
    }


def test_whole_candidates_keep_every_surface_variant() -> None:
    keys = extracted(
        'A conventional buck converter is shown.',
        'Two conventional buck converters are shown.',
    )
    assert keys['conventional buck converter'] == [
        'conventional buck converter',
        'conventional buck converters',
    ]


def test_stopword_modifiers_still_yield_their_piece() -> None:
    keys = extracted('The vehicle has front and rear wheels.')
    assert set(keys) == {'front wheel', 'rear wheel'}


def test_every_piece_surface_names_the_head() -> None:
    keys = extracted('The vehicle has front and rear wheels.')
    assert keys == {'front wheel': ['front wheels'], 'rear wheel': ['rear wheels']}


def test_tokens_between_the_conjuncts_and_the_head_enter_every_key() -> None:
    keys = extracted('The unit has residential or commercial power supply applications.')
    assert set(keys) == {
        'residential power supply application',
        'commercial power supply application',
    }


@pytest.mark.parametrize(
    ('sentence', 'phrase'),
    [
        (
            'The pad is a permanent or semi-permanent structure.',
            'permanent or semi-permanent structure',
        ),
        (
            'It is an eligibility (but not benefits) verification service.',
            'eligibility (but not benefits) verification service',
        ),
    ],
)
def test_hyphenated_words_and_parenthetical_contrasts_stay_whole(
    sentence: str, phrase: str
) -> None:
    doc: Doc = host()(sentence)
    start = sentence.index(phrase)
    assert JateDraw.pieces(doc, start, start + len(phrase), phrase, phrase) == (
        (phrase, phrase, start, start + len(phrase)),
    )
