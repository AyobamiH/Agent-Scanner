"""Pattern detection utilities for identifying AI and agentic code patterns.

Provides flexible keyword-based pattern matching for repository files and content.
Supports both substring and whole-word matching with configurable scoring weights.
Configuration is loaded from JSON files containing keyword lists and patterns.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from re import Pattern

logger = logging.getLogger(__name__)

# Regex pattern for sanitising text
NON_ALPHANUMERIC_PATTERN = r"[^a-z0-9]+"

_DEFAULT_KEYWORDS_PATH = Path(__file__).parent.parent / "config" / "keywords.json"


class PatternMatcher:
    """Flexible pattern matcher supporting path and content keyword matching.

    Implements two matching strategies for flexibility:
        - Substring matching: Detects keywords anywhere in text (e.g., "openai" matches "OpenAI")
        - Whole-word matching: Detects keywords as complete words only (e.g., "ai" doesn't match "await")

    Supports weighted scoring for prioritizing important patterns.
    Loads configuration from JSON files via from_file() class method.

    Key Methods:
        - match_path(path): Boolean check if file path matches any path patterns
        - match_content(content): Boolean check if content matches any content patterns
        - score_path(path): Weighted score (0+) for file path patterns
        - score_content(content): Weighted score (0+) for content patterns
        - get_matched_paths(content): List of matched keyword/phrase strings found in content
    """

    def __init__(
        self,
        content_keywords: Iterable[str] | None = None,
        content_whole_words: Iterable[str] | None = None,
        path_whole_words: Iterable[str] | None = None,
        prompt_phrase_patterns: dict[str, object] | None = None,
        min_sanitised_substring_length: int = 3,
    ) -> None:
        """Initialise the pattern matcher with keyword sets.

        Args:
            content_keywords: Keywords for substring matching in file contents.
            content_whole_words: Keywords for whole-word matching in file contents.
            path_whole_words: Keywords for whole-word matching in file paths.
            prompt_phrase_patterns: Configuration dict with phrase_patterns, strong_indicators, etc.
            min_sanitised_substring_length: Minimum length for sanitised substring matches.
        """
        self.content_keywords = [keyword.lower() for keyword in (content_keywords or [])]
        self.content_whole_words = [keyword.lower() for keyword in (content_whole_words or [])]
        self.path_whole_words = [keyword.lower() for keyword in (path_whole_words or [])]
        self.min_sanitised_substring_length = int(min_sanitised_substring_length or 3)

        self._prompt_phrase_config = prompt_phrase_patterns or {}
        raw_phrases = self._prompt_phrase_config.get("phrase_patterns", [])
        if isinstance(raw_phrases, list):
            self._phrase_patterns = [p.lower() for p in raw_phrases if isinstance(p, str)]
        else:
            self._phrase_patterns = []

        raw_weight = self._prompt_phrase_config.get("phrase_scoring_weight", 3)
        if isinstance(raw_weight, int):
            self._phrase_scoring_weight = raw_weight
        else:
            try:
                self._phrase_scoring_weight = int(str(raw_weight))
            except Exception:
                self._phrase_scoring_weight = 3
        self._phrase_patterns_re = self._build_whole_word_regex(self._phrase_patterns)

        self._content_whole_re = self._build_whole_word_regex(self.content_whole_words)
        self._path_whole_re = self._build_whole_word_regex(self.path_whole_words)

        self._tokenise_separator_re = re.compile(r"[-_./]")
        self._tokenise_camel_re = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
        self._tokenise_clean_re = re.compile(r"[^a-zA-Z0-9]+")

    @staticmethod
    def _build_whole_word_regex(words: list[str]) -> Pattern[str] | None:
        """Build a whole-word matching regex from a list of keywords.

        Args:
            words: List of keywords to match as whole words.

        Returns:
            Compiled regex pattern with word boundaries, or None if words is empty.
        """
        words = [word for word in words if len(word) >= 1]
        if not words:
            return None
        escaped = [re.escape(word) for word in words]
        return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)

    @classmethod
    def from_file(cls, path: str | None = None) -> PatternMatcher:
        """Create a PatternMatcher loading configuration from a JSON file.

        Args:
            path: Path to keywords configuration JSON file. If None, uses bundled
                  src/config/keywords.json.

        Returns:
            Configured PatternMatcher instance.

        Raises:
            ConfigurationError: If the configuration file exists but cannot be parsed.
        """
        from src.exceptions import ConfigurationError

        try:
            from src.detectors.keywords import load_keywords

            config_path = path or str(_DEFAULT_KEYWORDS_PATH)
            config = load_keywords(config_path)
        except FileNotFoundError:
            config = {
                "content_keywords": [],
                "content_whole_words": [],
                "path_whole_words": [],
            }
        except Exception as exc:
            raise ConfigurationError("Failed to load keywords configuration") from exc
        return cls(
            content_keywords=config.get("content_keywords", []),
            content_whole_words=config.get("content_whole_words", []),
            path_whole_words=config.get("path_whole_words", []),
            prompt_phrase_patterns=config.get("prompt_phrase_patterns", {}),
        )

    def match_path(self, path: str) -> bool:
        """Determine if a file path likely indicates AI or agentic content.

        Matching strategy:
            1. Check path whole-word regex against the lowercased path.
            2. Tokenise path and check for whole-word token matches.
            3. Check for substring matches against content keywords.

        Args:
            path: File path to evaluate.

        Returns:
            True if the path matches any pattern, False otherwise.
        """
        if not path:
            return False
        path_string = path
        path_lower = path_string.lower()
        if self._path_whole_re and self._path_whole_re.search(path_lower):
            logger.debug("Path whole-word match for %s", path)
            return True

        token_set = self._tokenise_text(path_string)
        for whole_word in self.path_whole_words:
            if whole_word in token_set:
                logger.debug("Path token match for %s in %s", whole_word, path)
                return True
            if whole_word.endswith("s"):
                singular = whole_word[:-1]
                if singular and singular in token_set:
                    logger.debug(
                        "Path token singular match for %s (from %s) in %s",
                        singular,
                        whole_word,
                        path,
                    )
                    return True
            else:
                plural = whole_word + "s"
                if plural in token_set:
                    logger.debug("Path token plural match for %s (as %s) in %s", whole_word, plural, path)
                    return True

        for keyword in self.content_keywords:
            if keyword in path_lower:
                logger.debug("Path substring match for %s", keyword)
                return True
        return False

    def score_path(self, path: str) -> int:
        """Calculate a weighted score for a file path.

        Whole-word matches contribute 2 points each, substring matches 1 point.

        Args:
            path: File path to score.

        Returns:
            Weighted score indicating strength of pattern match.
        """
        if not path:
            return 0
        path_lower = path.lower()
        score = 0
        if self._path_whole_re and self._path_whole_re.search(path_lower):
            score += 2
            logger.debug("Path whole-word regex matched for %s", path)
        token_set = self._tokenise_text(path)
        logger.debug("Path tokens for %s: %s", path, token_set)
        for whole_word in self.path_whole_words:
            if whole_word in token_set:
                score += 2
                logger.debug("Path token whole-word match '%s' in %s", whole_word, path)
            if whole_word.endswith("s"):
                singular = whole_word[:-1]
                if singular and singular in token_set:
                    score += 2
                    logger.debug(
                        "Path token singular match '%s' (from %s) in %s",
                        singular,
                        whole_word,
                        path,
                    )
            else:
                plural = whole_word + "s"
                if plural in token_set:
                    score += 2
                    logger.debug("Path token plural match '%s' (as %s) in %s", whole_word, plural, path)
        for keyword in self.content_keywords:
            if keyword in path_lower:
                score += 1
                logger.debug("Path substring keyword '%s' found in %s", keyword, path)
        return score

    def score_content(self, text: str) -> int:
        """Calculate a weighted score for file content.

        Whole-word matches contribute 2 points each, substring and sanitised matches
        contribute 1 point each.

        Args:
            text: File content to score.

        Returns:
            Weighted score indicating strength of pattern match.
        """
        if not text:
            return 0
        lower_text = text.lower()
        sanitised_text = re.sub(NON_ALPHANUMERIC_PATTERN, "", lower_text)
        token_set = self._tokenise_text(text)
        score = 0
        for keyword in self.content_keywords:
            if keyword in lower_text:
                score += 1
                logger.debug("Content substring keyword '%s' matched", keyword)
                continue
            if len(keyword) >= self.min_sanitised_substring_length:
                sanitised_keyword = re.sub(NON_ALPHANUMERIC_PATTERN, "", keyword)
                if sanitised_keyword and sanitised_keyword in sanitised_text:
                    score += 1
                    logger.debug(
                        "Content sanitised substring keyword '%s' matched as '%s'",
                        keyword,
                        sanitised_keyword,
                    )
                    continue
            if keyword in token_set:
                score += 1
                logger.debug("Content token keyword '%s' matched as token", keyword)

        whole_matches = set()
        if self._content_whole_re:
            matches = self._content_whole_re.findall(lower_text)
            if matches:
                logger.debug("Content whole-word regex matches: %s", matches)
            for match in matches:
                whole_matches.add(match.lower())
        for whole_word in self.content_whole_words:
            if whole_word in token_set:
                whole_matches.add(whole_word)
        if whole_matches:
            logger.debug("Content whole-word matches total: %s", whole_matches)
        score += 2 * len(whole_matches)
        return score

    def match_content(self, text: str) -> bool:
        """Determine if file content contains AI or agentic indicators.

        Strategy: performs cheap substring checks first, then whole-word regex matching.

        Args:
            text: File content to evaluate.

        Returns:
            True if the content matches any pattern, False otherwise.
        """
        if not text:
            return False
        lower_text = text.lower()
        sanitised_text = re.sub(NON_ALPHANUMERIC_PATTERN, "", lower_text)
        token_set = self._tokenise_text(text)
        for keyword in self.content_keywords:
            if keyword in lower_text:
                logger.debug("Content substring match for %s", keyword)
                return True
            if len(keyword) >= self.min_sanitised_substring_length:
                sanitised_keyword = re.sub(NON_ALPHANUMERIC_PATTERN, "", keyword)
                if sanitised_keyword and sanitised_keyword in sanitised_text:
                    logger.debug(
                        "Content sanitised substring match for %s (as %s)",
                        keyword,
                        sanitised_keyword,
                    )
                    return True
            if keyword in token_set:
                logger.debug("Content token match for %s", keyword)
                return True

        for whole_word in self.content_whole_words:
            if whole_word in token_set:
                logger.debug("Content whole-word token match for %s", whole_word)
                return True
        if self._content_whole_re and self._content_whole_re.search(lower_text):
            logger.debug("Content whole-word regex match")
            return True
        return False

    def contains_ai_keywords(self, text: str) -> bool:
        """Legacy alias for match_content.

        Provided for backward compatibility. New code should use match_content directly.

        Args:
            text: File content to evaluate.

        Returns:
            True if the content matches any pattern, False otherwise.
        """
        return self.match_content(text)

    def score_prompt_phrases(self, text: str) -> int:
        """Calculate a weighted score for detected prompt phrases in text.

        Prompt phrases like "You are an expert..." are strong indicators of agent
        definitions, particularly when found in system_prompt or instruction fields.

        Args:
            text: File content or field value to score.

        Returns:
            Weighted score indicating presence of agentic prompt language.
            Each matched phrase contributes phrase_scoring_weight points.
        """
        if not text or not self._phrase_patterns:
            return 0

        lower_text = text.lower()
        score = 0
        matched_phrases = set()

        if self._phrase_patterns_re:
            matches = self._phrase_patterns_re.findall(lower_text)
            if matches:
                logger.debug("Prompt phrase regex matches: %s", matches)
                for match in matches:
                    matched_phrases.add(match.lower())

        for phrase in self._phrase_patterns:
            if phrase in lower_text and phrase not in matched_phrases:
                logger.debug("Prompt phrase substring match: '%s'", phrase)
                matched_phrases.add(phrase)

        if matched_phrases:
            score = len(matched_phrases) * self._phrase_scoring_weight
            logger.debug(
                "Prompt phrase scoring: found %d phrases for %d points",
                len(matched_phrases),
                score,
            )

        return score

    def match_prompt_phrases(self, text: str) -> bool:
        """Determine if text contains agentic prompt phrases.

        Args:
            text: File content or field value to evaluate.

        Returns:
            True if any prompt phrase pattern is found, False otherwise.
        """
        if not text or not self._phrase_patterns:
            return False

        lower_text = text.lower()

        if self._phrase_patterns_re and self._phrase_patterns_re.search(lower_text):
            logger.debug("Prompt phrase regex match found")
            return True

        for phrase in self._phrase_patterns:
            if phrase in lower_text:
                logger.debug("Prompt phrase match found: '%s'", phrase)
                return True

        return False

    def _tokenise_text(self, text: str) -> set[str]:
        """Extract normalised tokens from text for pattern matching.

        Handles camelCase, snake_case, kebab-case, dotted notation, and concatenated
        forms. Tokens are lowercased with non-alphanumeric characters removed.

        Args:
            text: Raw text or identifier to tokenise.

        Returns:
            Set of normalised token strings.
        """
        if not text:
            return set()

        normalised = self._tokenise_separator_re.sub(" ", text)
        parts = normalised.split()
        tokens = set()
        for part in parts:
            if not part:
                continue
            camel_split = self._tokenise_camel_re.sub(" ", part)
            for sub_part in camel_split.split():
                cleaned = self._tokenise_clean_re.sub("", sub_part).lower()
                if cleaned:
                    tokens.add(cleaned)
        return tokens
