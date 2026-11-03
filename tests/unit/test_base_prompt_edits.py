"""Tests for the pure edit-list helpers behind PromptComposer's base prompt.

``parse_edit_list`` reads the LLM reply, ``apply_edits`` applies it to the
base prompt. The guarantee under test: text no edit touches comes out
character for character, and an edit that cannot be placed is counted, not
guessed.

Mocks: none — both functions are pure.
"""

from __future__ import annotations

import pytest

from comfyui_prompt_tools.base_prompt_edits import (
    EditListError,
    apply_edits,
    locate,
    parse_edit_list,
)

pytestmark = pytest.mark.unit

PROSE = (
    "a woman with long red curly hair in a yellow summer dress walking "
    "through a foggy forest at dawn, 35mm film"
)
TAGS = "1girl, red hair, yellow dress, long hair, forest, dawn"


def rep(find, replace):
    return {"find": find, "replace": replace}


def add(after, insert):
    return {"after": after, "insert": insert}


# ---------------------------------------------------------------------------
# Replacements
# ---------------------------------------------------------------------------
class TestReplacements:
    def test_briefing_example(self):
        result = apply_edits(PROSE, [rep("long red curly hair", "short platinum bob")], [])
        assert result.text == (
            "a woman with short platinum bob in a yellow summer dress walking "
            "through a foggy forest at dawn, 35mm film"
        )
        assert (result.replaced, result.added, result.skipped) == (1, 0, 0)

    def test_untouched_text_is_identical(self):
        result = apply_edits(PROSE, [rep("yellow summer dress", "black coat")], [])
        before, after = PROSE.split("yellow summer dress")
        assert result.text == f"{before}black coat{after}"

    def test_tolerant_match_ignores_case_and_whitespace_runs(self):
        result = apply_edits(PROSE, [rep("Long  RED\ncurly hair", "a bob")], [])
        assert result.text.startswith("a woman with a bob in a yellow")
        assert result.replaced == 1

    def test_tolerance_is_not_fuzzy(self):
        """A paraphrased find is skipped, never approximated."""
        result = apply_edits(PROSE, [rep("long red curled hair", "a bob")], [])
        assert result.text == PROSE
        assert result.not_found == 1

    def test_not_found_is_skipped_and_counted(self):
        result = apply_edits(
            PROSE,
            [rep("blue eyes", "green eyes"), rep("35mm film", "digital photo")],
            [],
        )
        assert result.text.endswith("at dawn, digital photo")
        assert (result.replaced, result.not_found) == (1, 1)

    def test_only_first_occurrence_is_replaced(self):
        result = apply_edits("red hair, red dress", [rep("red", "blue")], [])
        assert result.text == "blue hair, red dress"

    def test_overlapping_later_replacement_is_dropped(self):
        result = apply_edits(
            PROSE,
            [rep("red curly hair", "a bob"), rep("long red", "short black")],
            [],
        )
        assert "long a bob" in result.text
        assert "short black" not in result.text
        assert (result.replaced, result.overlap) == (1, 1)

    def test_adjacent_replacements_do_not_overlap(self):
        result = apply_edits("ab", [rep("a", "X"), rep("b", "Y")], [])
        assert result.text == "XY"
        assert result.overlap == 0

    def test_positions_refer_to_the_original(self):
        """A longer replacement does not shift the next match."""
        result = apply_edits(
            TAGS,
            [rep("red hair", "very long platinum blonde hair"), rep("forest", "beach")],
            [],
        )
        assert result.text == (
            "1girl, very long platinum blonde hair, yellow dress, long hair, beach, dawn"
        )

    def test_replacement_is_not_searched_in_new_text(self):
        """A find that only exists inside another replacement is not found."""
        result = apply_edits("a cat", [rep("cat", "red dog"), rep("red dog", "x")], [])
        assert result.text == "a red dog"
        assert result.not_found == 1

    def test_whitespace_framing_the_match_is_kept(self):
        result = apply_edits("a  b  c", [rep(" b ", "X")], [])
        assert result.text == "a  X  c"

    def test_replace_text_is_stripped(self):
        result = apply_edits(TAGS, [rep("forest", "  beach  ")], [])
        assert ", beach, dawn" in result.text


class TestDeletions:
    def test_middle_tag_takes_its_following_separator(self):
        result = apply_edits(TAGS, [rep("long hair", "")], [])
        assert result.text == "1girl, red hair, yellow dress, forest, dawn"

    def test_last_tag_takes_its_preceding_separator(self):
        result = apply_edits(TAGS, [rep("dawn", "")], [])
        assert result.text == "1girl, red hair, yellow dress, long hair, forest"

    def test_first_tag(self):
        result = apply_edits(TAGS, [rep("1girl", "")], [])
        assert result.text == "red hair, yellow dress, long hair, forest, dawn"

    def test_prose_deletion_drops_one_space(self):
        result = apply_edits("a woman with red hair walking", [rep("with red hair", "")], [])
        assert result.text == "a woman walking"

    def test_prose_deletion_at_the_end(self):
        result = apply_edits("a woman walking slowly", [rep("slowly", "")], [])
        assert result.text == "a woman walking"

    def test_split_feature_replace_first_delete_rest(self):
        result = apply_edits(
            TAGS, [rep("red hair", "platinum bob"), rep("long hair", "")], []
        )
        assert result.text == "1girl, platinum bob, yellow dress, forest, dawn"
        assert result.replaced == 2

    def test_two_deletions_next_to_each_other(self):
        result = apply_edits(TAGS, [rep("long hair", ""), rep("forest", "")], [])
        assert result.text == "1girl, red hair, yellow dress, dawn"

    def test_deletion_before_last_deleted_tag(self):
        result = apply_edits(TAGS, [rep("forest", ""), rep("dawn", "")], [])
        assert result.text == "1girl, red hair, yellow dress, long hair"


# ---------------------------------------------------------------------------
# Additions
# ---------------------------------------------------------------------------
class TestAdditions:
    def test_with_anchor_joins_with_space(self):
        result = apply_edits(PROSE, [], [add("summer dress", "and silver earrings")])
        assert "yellow summer dress and silver earrings walking" in result.text
        assert result.added == 1

    def test_with_anchor_and_leading_comma(self):
        result = apply_edits(TAGS, [], [add("yellow dress", ", silver earrings")])
        assert result.text == (
            "1girl, red hair, yellow dress, silver earrings, long hair, forest, dawn"
        )

    def test_without_anchor_appends_with_comma(self):
        result = apply_edits(PROSE, [], [add("", "soft film grain")])
        assert result.text == PROSE + ", soft film grain"

    def test_missing_after_key_appends(self):
        result = apply_edits(TAGS, [], [{"insert": "rim light"}])
        assert result.text == TAGS + ", rim light"

    def test_null_after_appends(self):
        result = apply_edits(TAGS, [], [add(None, "rim light")])
        assert result.text == TAGS + ", rim light"

    def test_unknown_anchor_appends(self):
        """The feature is still wanted; only its position is unknown."""
        result = apply_edits(TAGS, [], [add("no such text", "rim light")])
        assert result.text == TAGS + ", rim light"
        assert (result.added, result.skipped) == (1, 0)

    def test_append_after_sentence_end_uses_space(self):
        result = apply_edits("A woman walks.", [], [add("", "Rain falls.")])
        assert result.text == "A woman walks. Rain falls."

    def test_append_with_leading_punctuation(self):
        result = apply_edits("a, b", [], [add("", ", c")])
        assert result.text == "a, b, c"

    def test_anchor_on_replaced_text_follows_the_replacement(self):
        result = apply_edits(
            PROSE,
            [rep("long red curly hair", "a short bob")],
            [add("long red curly hair", "and green eyes")],
        )
        assert result.text.startswith("a woman with a short bob and green eyes in a")

    def test_anchor_ending_inside_a_replacement_moves_behind_it(self):
        result = apply_edits(
            PROSE,
            [rep("long red curly hair", "a short bob")],
            [add("long red", "and green eyes")],
        )
        assert result.text.startswith("a woman with a short bob and green eyes in a")

    def test_several_additions_keep_list_order(self):
        result = apply_edits(
            TAGS, [], [add("forest", ", fog"), add("forest", ", moss")]
        )
        assert "forest, fog, moss, dawn" in result.text

    def test_addition_before_following_word_gets_a_space(self):
        result = apply_edits("ab cd", [], [add("ab", "X")])
        assert result.text == "ab X cd"


# ---------------------------------------------------------------------------
# Empty and malformed input
# ---------------------------------------------------------------------------
class TestEmptyAndInvalid:
    def test_empty_lists_return_base_unchanged(self):
        result = apply_edits(PROSE, [], [])
        assert result.text == PROSE
        assert (result.replaced, result.added, result.skipped) == (0, 0, 0)

    def test_none_lists_return_base_unchanged(self):
        assert apply_edits(PROSE, None, None).text == PROSE

    @pytest.mark.parametrize(
        "entry",
        [
            "a string",
            {"replace": "x"},
            {"find": "", "replace": "x"},
            {"find": "   ", "replace": "x"},
            {"find": 3, "replace": "x"},
            {"find": "forest", "replace": None},
            {"find": "forest", "replace": 5},
        ],
    )
    def test_malformed_replacement_is_invalid(self, entry):
        result = apply_edits(TAGS, [entry], [])
        assert result.text == TAGS
        assert result.invalid == 1

    @pytest.mark.parametrize(
        "entry",
        [
            ["list"],
            {"after": "forest"},
            {"after": "forest", "insert": ""},
            {"after": "forest", "insert": 7},
            {"after": 7, "insert": "fog"},
        ],
    )
    def test_malformed_addition_is_invalid(self, entry):
        result = apply_edits(TAGS, [], [entry])
        assert result.text == TAGS
        assert result.invalid == 1

    def test_missing_replace_key_means_delete(self):
        result = apply_edits(TAGS, [{"find": "dawn"}], [])
        assert result.text == "1girl, red hair, yellow dress, long hair, forest"

    def test_skipped_sums_all_reasons(self):
        result = apply_edits(
            PROSE,
            [rep("nope", "x"), rep("red curly hair", "y"), rep("long red", "z"), "bad"],
            [],
        )
        assert (result.not_found, result.overlap, result.invalid) == (1, 1, 1)
        assert result.skipped == 3


class TestLocate:
    def test_exact(self):
        assert locate("abc def", "def") == (4, 7)

    def test_tolerant(self):
        assert locate("abc   DEF", "abc def") == (0, 9)

    def test_empty_needle(self):
        assert locate("abc", "") is None
        assert locate("abc", "   ") is None

    def test_regex_characters_are_literal(self):
        assert locate("(masterpiece:1.2), 1girl", "(MASTERPIECE:1.2)") == (0, 17)


# ---------------------------------------------------------------------------
# Reading the LLM reply
# ---------------------------------------------------------------------------
VALID = '{"replacements": [{"find": "a", "replace": "b"}], "additions": []}'


class TestParseEditList:
    def test_plain_json(self):
        assert parse_edit_list(VALID) == ([{"find": "a", "replace": "b"}], [])

    def test_code_fence(self):
        assert parse_edit_list(f"```json\n{VALID}\n```")[0] == [
            {"find": "a", "replace": "b"}
        ]

    def test_code_fence_without_language(self):
        assert parse_edit_list(f"```\n{VALID}\n```")[0][0]["find"] == "a"

    def test_think_block(self):
        reply = '<think>maybe {"replacements": [{"find": "x"}]}</think>\n' + VALID
        assert parse_edit_list(reply)[0][0]["find"] == "a"

    def test_orphan_closing_think_tag(self):
        """A reasoning parser may strip the opening tag only."""
        reply = 'thinking {"additions": [{"insert": "x"}]} </think>' + VALID
        assert parse_edit_list(reply)[0][0]["find"] == "a"

    def test_unterminated_think_after_the_object(self):
        assert parse_edit_list(VALID + "\n<think>cut off")[0][0]["find"] == "a"

    def test_surrounding_prose(self):
        reply = f"Here is the edit list:\n{VALID}\nHope this helps!"
        assert parse_edit_list(reply)[0][0]["find"] == "a"

    def test_braces_in_prose_before_the_object(self):
        reply = "Note {this} first. " + VALID
        assert parse_edit_list(reply)[0][0]["find"] == "a"

    def test_missing_key_is_empty_list(self):
        assert parse_edit_list('{"additions": [{"insert": "x"}]}') == (
            [],
            [{"insert": "x"}],
        )

    def test_null_lists_are_empty(self):
        assert parse_edit_list('{"replacements": null, "additions": null}') == ([], [])

    def test_empty_lists(self):
        assert parse_edit_list('{"replacements": [], "additions": []}') == ([], [])

    @pytest.mark.parametrize(
        "reply",
        [
            "",
            "   ",
            "a woman with a short bob in a yellow dress",
            '{"replacements": [{"find": "a", "replace": "b"}',
            '{"something": "else"}',
            "[1, 2, 3]",
            "<think>only reasoning, never finished",
        ],
    )
    def test_unreadable_reply_raises(self, reply):
        with pytest.raises(EditListError):
            parse_edit_list(reply)

    @pytest.mark.parametrize(
        "reply",
        [
            '{"replacements": "find x", "additions": []}',
            '{"replacements": [], "additions": {"insert": "x"}}',
        ],
    )
    def test_non_list_section_raises(self, reply):
        with pytest.raises(EditListError):
            parse_edit_list(reply)

    def test_error_is_a_value_error(self):
        assert issubclass(EditListError, ValueError)
