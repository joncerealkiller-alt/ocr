"""
Constrained decoding: a transformers LogitsProcessor that forces
generation to only pick tokens whose decoded text is entirely made of
"expected" characters - built to prevent off-script token fallback
(e.g. a stray Cyrillic fragment like "Малы" appearing where English
handwriting was expected), confirmed as a real failure mode across
multiple test runs this session (see recent git log / data/debug_model_inputs
captures) under low visual confidence.

WHY THIS RECOVERS ANSWERS THE EXISTING SCRUB CANNOT (2026-07-24):
core/row_extraction.py's _scrub_example_leakage() (called from
parse_row_output()) is a POST-HOC safety net - it can only detect
already-generated bad output and force it to "?"/unclear. It cannot
un-generate a wrong answer. Masking disallowed tokens at generation
time is different in kind: the model is still free to pick whatever
IT considers most likely among tokens that decode to expected-script
text, so a correct English reading that was merely LESS likely than a
wrong-script fallback token can still be produced. This is additive to
the existing scrub, not a replacement for it - a model can still
legitimately produce a wrong ENGLISH answer, which this mechanism has
no way to catch (that's still the scrub's job).

SCOPE (2026-07-24, confirmed via a real audit of every core/loaders/*.py
file's own _run_generate(), not assumed):
  - QUALIFIES (calls self.model.generate(**inputs, **gen_kwargs) directly,
    a real transformers logits_processor= kwarg path): FlorenceLoader,
    GemmaLoader, GlmOcrLoader, GotOcr2Loader, GraniteVisionLoader,
    InternVLLoader, OlmOcrLoader, PixtralLoader, Qwen3VLLoader, QwenLoader,
    SmolVLM2Loader - 11 loaders, wired via BaseLoader._maybe_add_charset_
    logits_processor() (see that method's docstring for how each loader
    calls it).
  - DOES NOT QUALIFY - ChandraLoader: calls the external chandra-ocr
    package's own generate_hf(batch, self.model) - no logits_processor
    kwarg is exposed at that call site at all (opaque to this project).
  - DOES NOT QUALIFY (for now) - MoondreamLoader: runs in a separate
    venv via SubprocessLoaderBase, and its public query()/caption() API
    only takes max_tokens/temperature/top_p/adapter/model - nothing
    token-level. CONFIRMED POSSIBLE if ever needed (2026-07-24 real
    investigation, not assumed): moondream2 runs fully in-process inside
    its own worker (not a cloud API - that's moondream3-preview, a
    DIFFERENT model this project doesn't use), and doesn't even call
    transformers' generate() - it's a fully hand-rolled decode loop
    (MoondreamModel._generate_answer() in the cached snapshot's
    moondream.py) that already masks specific token ids to -inf before
    sampling (e.g. logits_BV[:, self.config.tokenizer.answer_id] =
    float("-inf")) - adding a charset mask there would be a small,
    mechanically similar addition. NOT built here because (a) it
    requires monkeypatching a PRIVATE, undocumented method inside
    moondream2's own trust_remote_code - fundamentally less stable than
    the public, versioned transformers logits_processor= kwarg every
    other loader uses, and (b) no real Cyrillic-fallback (or other off-
    script) failure has actually been observed from moondream2 in this
    project so far - the observed cases were other VLMs. Revisit only
    if real evidence of the same failure mode shows up for moondream2
    specifically.

DEFAULT ALLOWED CHARACTER SET - deliberately NOT bare ASCII (2026-07-24
design decision, not what was originally specified): this project's
data is Western/European-origin genealogical records - personal and
place names routinely need accented Latin letters ("François",
"Québec", "José") and are not exotic edge cases here. An ASCII-only
mask would actively break correct transcription of those names, trading
one failure mode (off-script fallback) for another (mangled real
answers) rather than fixing the actual problem. default_allowed_char()
therefore allows the full Latin alphabet (bare + accented, i.e. Basic
Latin + Latin-1 Supplement + Latin Extended-A/B letter ranges), ASCII
digits, and the punctuation this project's own row/document prompts
actually use (see build_row_prompt/build_structuring_prompt in
core/row_extraction.py for the "ColumnName: value|confidence" format,
and core/extraction_parsing.py for the document-level pipe-delimited
convention) - colon, pipe, question mark, comma, period, hyphen,
straight/curly quotes/apostrophes, whitespace (space/tab/newline - each
column is its own output line). Cyrillic, CJK, Arabic, Hebrew, etc. are
NOT in any allowed range, so the actually-observed failure mode (a
stray Cyrillic fragment) is blocked, without touching legitimate
accented-Latin content.

Also includes `{}[]` - NOT part of the row/document pipe-delimited
format itself, but required by OlmOcrLoader's raw output (genuine JSON,
parsed via json.loads() - see that loader's _run_generate) and
FlorenceLoader's json.dumps()-wrapped output (see that loader's
docstring). Costs nothing against the actual anti-Cyrillic goal (these
are structural brackets, not script characters a wrong-script fallback
would ever produce) while keeping both loaders' real output format
representable under the mask.
"""

from __future__ import annotations

from typing import Callable

import torch
from transformers import LogitsProcessor


# Whitespace/structural punctuation this project's own prompts require -
# see module docstring for exactly where each character is used. Underscore
# added 2026-07-26 (enabling restrict_output_charset for GemmaLoader's
# classification role, config/models/gemma.yaml): its expected field-name
# labels - document_type, personal_names, place_names, visible_dates,
# subject_keywords (core/extraction_parsing.py's _KNOWN_FIELDS) - all
# require it; without this, masking would have actively broken Gemma's
# own classification output format, not just blocked off-script fallback.
_ALLOWED_PUNCTUATION = set(" \t\n:|?,.-_'\"“”‘’{}[]")

# (start, end) INCLUSIVE Unicode codepoint ranges treated as "Latin
# letters" for this predicate - Basic Latin (bare a-z/A-Z, handled via
# str.isalpha() below since the range also contains digits/punctuation
# already covered separately), Latin-1 Supplement, Latin Extended-A,
# Latin Extended-B. Deliberately excludes Cyrillic (0400-04FF), CJK
# (4E00-9FFF and friends), Arabic (0600-06FF), Hebrew (0590-05FF), etc.
_LATIN_LETTER_RANGES = (
    (0x0041, 0x005A),  # A-Z
    (0x0061, 0x007A),  # a-z
    (0x00C0, 0x024F),  # Latin-1 Supplement + Latin Extended-A/B letters
)


def default_allowed_char(ch: str) -> bool:
    """
    True if `ch` is a Latin letter (bare or accented), an ASCII digit,
    or one of this project's own structural/punctuation characters -
    see module docstring for the reasoning behind not using bare ASCII.
    """
    if ch in _ALLOWED_PUNCTUATION:
        return True
    if "0" <= ch <= "9":
        return True
    cp = ord(ch)
    for lo, hi in _LATIN_LETTER_RANGES:
        if lo <= cp <= hi:
            # Latin-1 Supplement/Extended-A/B ranges include a handful
            # of non-letter symbols too (e.g. U+00D7 MULTIPLICATION
            # SIGN) - isalpha() excludes those, bare A-Z/a-z is already
            # guaranteed alphabetic so this is a cheap no-op there.
            return ch.isalpha()
    return False


def _tokenizer_cache_key(tokenizer) -> tuple:
    """
    Identifies a tokenizer for mask caching without hashing its full
    vocab. Prefers `name_or_path` (stable across separate loader
    instances that load the SAME model repo - e.g. two-stage
    extraction's sequential model loads - so the mask is computed once
    per distinct model, not once per loader instance) plus vocab length
    as a cheap sanity check against two different tokenizers sharing a
    name. Falls back to raw object identity if name_or_path is missing
    (e.g. some custom tokenizers) - safe, just narrower reuse.
    """
    name = getattr(tokenizer, "name_or_path", None)
    if name:
        return (name, len(tokenizer))
    return (id(tokenizer), len(tokenizer))


# Module-level cache: computing the mask means decoding every token id
# in the vocab once (tens of thousands of tokenizer.decode() calls) -
# real cost, see AllowedCharsLogitsProcessor's docstring for measured
# numbers. Cached so this only happens once per distinct tokenizer, not
# once per generation call (which would make every single _run_generate
# call pay this cost).
_MASK_CACHE: dict[tuple, torch.Tensor] = {}


def get_allowed_token_mask(
    tokenizer, allowed_char_predicate: Callable[[str], bool] = default_allowed_char,
) -> torch.Tensor:
    """
    Returns a cached boolean tensor of length len(tokenizer): True where
    that token id is allowed. Special tokens (EOS/BOS/PAD/and every
    other id in tokenizer.all_special_ids) are ALWAYS allowed regardless
    of decoded text or the predicate - masking EOS in particular would
    make generation unable to ever stop cleanly, and masking PAD/BOS
    could break batching/prompting entirely. This is the one hard rule
    this function does not let allowed_char_predicate override.
    """
    cache_key = (_tokenizer_cache_key(tokenizer), id(allowed_char_predicate))
    cached = _MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    vocab_size = len(tokenizer)
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    special_ids = set(tokenizer.all_special_ids)

    for token_id in range(vocab_size):
        if token_id in special_ids:
            mask[token_id] = True
            continue
        try:
            text = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            # Fail OPEN, not closed, for a token id this tokenizer
            # itself can't decode - an unexpected decode error is a
            # tokenizer-internals surprise, not evidence the token is
            # off-script; refusing to ever emit it could break
            # generation in a way that has nothing to do with this
            # feature's actual purpose.
            mask[token_id] = True
            continue
        # all() over an empty string's characters is True already, so
        # a token that decodes to "" (e.g. a formatting/continuation
        # piece not already in special_ids) is allowed with no special
        # case needed here.
        if all(allowed_char_predicate(c) for c in text):
            mask[token_id] = True

    _MASK_CACHE[cache_key] = mask
    return mask


class AllowedCharsLogitsProcessor(LogitsProcessor):
    """
    Sets logits for every disallowed token id to -inf before each
    sampling/argmax step, so generate() can never pick them - the
    model's own probability ranking among ALLOWED tokens is otherwise
    untouched (this is a hard mask, not a soft penalty like
    repetition_penalty).

    Usage (see BaseLoader._maybe_add_charset_logits_processor(), the
    actual wiring point every qualifying loader calls):
        gen_kwargs["logits_processor"] = [
            AllowedCharsLogitsProcessor(self.tokenizer)
        ]
        self.model.generate(**inputs, **gen_kwargs)

    Cost (measured 2026-07-24 against SmolVLM2-2.2B-Instruct's real
    tokenizer, 49,280 tokens): mask computation is a real but ONE-TIME
    cost (tens of thousands of individual tokenizer.decode() calls) -
    see get_allowed_token_mask()'s module-level cache. Per-step cost
    once the mask exists is a single masked_fill over the vocab
    dimension - negligible next to one forward pass of a multi-billion-
    parameter model.
    """

    def __init__(
        self, tokenizer, allowed_char_predicate: Callable[[str], bool] = default_allowed_char,
    ):
        self.mask = get_allowed_token_mask(tokenizer, allowed_char_predicate)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = self.mask.to(scores.device)
        vocab_dim = scores.shape[-1]
        # len(tokenizer) and the model's actual lm_head output dimension
        # are NOT always equal - real bug found 2026-07-24 via a live
        # InternVL run: "The size of tensor a (151675) must match the
        # size of tensor b (151674)", a crash on every single call once
        # this feature was turned on. Two known directions, both must be
        # handled, not just one:
        #   - scores WIDER than the mask: some models pad the lm_head to
        #     a rounder number for hardware efficiency (e.g. a multiple
        #     of 64) - pad the mask with True (allowed) for the extra
        #     positions, which are never real vocabulary the tokenizer
        #     would ever ask to emit, so leaving them unmasked is
        #     harmless.
        #   - scores NARROWER than the mask (InternVL's actual case
        #     above - len(tokenizer)=151675 > the model's real output
        #     dim=151674, a genuine tokenizer/model vocab-size mismatch,
        #     not a hardware-padding one): truncate the mask down to
        #     vocab_dim. Safe, not a data-loss workaround - the model
        #     CANNOT produce logits for a token id past its own output
        #     layer's width regardless of what this mask says, so any
        #     mask entries beyond vocab_dim were already meaningless.
        if mask.shape[0] < vocab_dim:
            pad = torch.ones(vocab_dim - mask.shape[0], dtype=torch.bool, device=mask.device)
            mask = torch.cat([mask, pad])
        elif mask.shape[0] > vocab_dim:
            mask = mask[:vocab_dim]
        return scores.masked_fill(~mask, float("-inf"))
