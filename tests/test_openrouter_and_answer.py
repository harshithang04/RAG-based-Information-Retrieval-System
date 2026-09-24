"""
Comprehensive offline tests for openrouter_client and RAG pipeline.
All network calls are mocked; no real API keys or models are used.
"""
import base64
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import numpy as np
import pytest

import app
import config
import openrouter_client
from rag_pipeline import RAGPipeline


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def fake_response():
    """Create a mock response object for requests.post."""
    def _make_response(
        status_code=200,
        json_data=None,
        text="",
    ):
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = text
        resp.json = MagicMock(return_value=json_data or {})
        return resp
    return _make_response


@pytest.fixture
def fake_embedder():
    """Create a mock embedder with encode method."""
    embedder = MagicMock()
    # Return dummy embeddings (random vectors)
    embedder.encode = MagicMock(
        side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 384).astype(np.float32)
    )
    return embedder


@pytest.fixture
def mock_openrouter_post(monkeypatch, fake_response):
    """Monkeypatch requests.post for OpenRouter calls."""
    posted = []

    def capture_post(url, headers=None, json=None, timeout=None):
        posted.append({
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        })
        # Default success response
        return fake_response(
            status_code=200,
            json_data={
                "choices": [
                    {
                        "message": {
                            "content": "Mock response text",
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("requests.post", capture_post)
    return posted


# ============================================================================
# Tests: openrouter_client.chat_completion
# ============================================================================


class TestChatCompletionErrorHandling:
    """Test error conditions in chat_completion."""

    def test_empty_api_key_raises_before_request(self, monkeypatch, mock_openrouter_post):
        """Empty API key should raise OpenRouterError before making any request."""
        with pytest.raises(openrouter_client.OpenRouterError, match="No OpenRouter API key"):
            openrouter_client.chat_completion([], "model", api_key="")
        assert len(mock_openrouter_post) == 0

    def test_none_api_key_raises_before_request(self, monkeypatch, mock_openrouter_post):
        """None API key should raise OpenRouterError before making any request."""
        # The _headers function expects a string, so test with empty string which is falsy
        with pytest.raises(openrouter_client.OpenRouterError):
            openrouter_client.chat_completion([], "model", api_key=None)
        assert len(mock_openrouter_post) == 0

    def test_non_200_status_raises_with_status_and_body(self, monkeypatch, fake_response):
        """Non-200 status code should raise with status code and body."""
        body_text = "Rate limited" * 100  # Create a longer body to test truncation

        def fake_post(*args, **kwargs):
            return fake_response(status_code=429, text=body_text)

        monkeypatch.setattr("requests.post", fake_post)

        with pytest.raises(openrouter_client.OpenRouterError) as exc_info:
            openrouter_client.chat_completion([], "model", api_key="fake_key")

        err_msg = str(exc_info.value)
        assert "429" in err_msg
        assert "Rate limited" in err_msg
        # Body should be truncated to 500 chars
        assert len(err_msg) <= 550  # 500 + some buffer for other text

    def test_missing_choices_in_response_raises(self, monkeypatch, fake_response):
        """Response with no 'choices' key should raise OpenRouterError."""
        def fake_post(*args, **kwargs):
            return fake_response(status_code=200, json_data={"result": "no choices"})

        monkeypatch.setattr("requests.post", fake_post)

        with pytest.raises(openrouter_client.OpenRouterError, match="Unexpected OpenRouter response"):
            openrouter_client.chat_completion([], "model", api_key="fake_key")

    def test_empty_choices_list_raises(self, monkeypatch, fake_response):
        """Response with empty 'choices' list should raise OpenRouterError."""
        def fake_post(*args, **kwargs):
            return fake_response(status_code=200, json_data={"choices": []})

        monkeypatch.setattr("requests.post", fake_post)

        with pytest.raises(openrouter_client.OpenRouterError, match="Unexpected OpenRouter response"):
            openrouter_client.chat_completion([], "model", api_key="fake_key")

    def test_null_content_with_reasoning_raises_thinking_budget_error(self, monkeypatch, fake_response):
        """Content=None with reasoning field should raise specific 'thinking budget' error."""
        def fake_post(*args, **kwargs):
            return fake_response(
                status_code=200,
                json_data={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "reasoning": "The model was thinking...",
                            }
                        }
                    ]
                },
            )

        monkeypatch.setattr("requests.post", fake_post)

        with pytest.raises(openrouter_client.OpenRouterError) as exc_info:
            openrouter_client.chat_completion([], "model", api_key="fake_key")

        assert "thinking" in str(exc_info.value).lower() or "token budget" in str(exc_info.value).lower()

    def test_null_content_without_reasoning_raises_generic_error(self, monkeypatch, fake_response):
        """Content=None without reasoning should raise generic 'Unexpected' error."""
        def fake_post(*args, **kwargs):
            return fake_response(
                status_code=200,
                json_data={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                            }
                        }
                    ]
                },
            )

        monkeypatch.setattr("requests.post", fake_post)

        with pytest.raises(openrouter_client.OpenRouterError, match="Unexpected OpenRouter response"):
            openrouter_client.chat_completion([], "model", api_key="fake_key")


class TestChatCompletionSuccessCases:
    """Test successful chat_completion calls."""

    def test_content_whitespace_is_stripped(self, monkeypatch, fake_response):
        """Response content should be stripped of leading/trailing whitespace."""
        def fake_post(*args, **kwargs):
            return fake_response(
                status_code=200,
                json_data={
                    "choices": [
                        {
                            "message": {
                                "content": "  \n  Response text  \t",
                            }
                        }
                    ]
                },
            )

        monkeypatch.setattr("requests.post", fake_post)

        result = openrouter_client.chat_completion([], "model", api_key="fake_key")
        assert result == "Response text"

    def test_payload_includes_required_fields(self, monkeypatch, mock_openrouter_post, fake_response):
        """Payload should include model, temperature, max_tokens."""
        def fake_post(url, headers=None, json=None, timeout=None):
            mock_openrouter_post.append({
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            })
            return fake_response(
                status_code=200,
                json_data={
                    "choices": [{"message": {"content": "ok"}}]
                },
            )

        monkeypatch.setattr("requests.post", fake_post)

        openrouter_client.chat_completion(
            [],
            model="test-model",
            api_key="key",
            temperature=0.5,
            max_tokens=100,
        )

        post_call = mock_openrouter_post[0]
        payload = post_call["json"]
        assert payload["model"] == "test-model"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100
        assert "messages" in payload

    def test_top_p_only_in_payload_when_not_none(self, monkeypatch, mock_openrouter_post, fake_response):
        """top_p should only be in payload when not None."""
        def fake_post(url, headers=None, json=None, timeout=None):
            mock_openrouter_post.append({"json": json})
            return fake_response(200, {"choices": [{"message": {"content": "ok"}}]})

        monkeypatch.setattr("requests.post", fake_post)

        # Test with top_p=None
        openrouter_client.chat_completion([], "model", "key", top_p=None)
        assert "top_p" not in mock_openrouter_post[0]["json"]

        # Test with top_p=0.9
        openrouter_client.chat_completion([], "model", "key", top_p=0.9)
        assert mock_openrouter_post[1]["json"]["top_p"] == 0.9

    def test_top_k_only_in_payload_when_not_none(self, monkeypatch, mock_openrouter_post, fake_response):
        """top_k should only be in payload when not None."""
        def fake_post(url, headers=None, json=None, timeout=None):
            mock_openrouter_post.append({"json": json})
            return fake_response(200, {"choices": [{"message": {"content": "ok"}}]})

        monkeypatch.setattr("requests.post", fake_post)

        # Test with top_k=None
        openrouter_client.chat_completion([], "model", "key", top_k=None)
        assert "top_k" not in mock_openrouter_post[0]["json"]

        # Test with top_k=40
        openrouter_client.chat_completion([], "model", "key", top_k=40)
        assert mock_openrouter_post[1]["json"]["top_k"] == 40

    def test_authorization_header_format(self, monkeypatch, mock_openrouter_post, fake_response):
        """Authorization header should be 'Bearer <key>'."""
        def fake_post(url, headers=None, json=None, timeout=None):
            mock_openrouter_post.append({"headers": headers})
            return fake_response(200, {"choices": [{"message": {"content": "ok"}}]})

        monkeypatch.setattr("requests.post", fake_post)

        openrouter_client.chat_completion([], "model", "test_key_123")

        headers = mock_openrouter_post[0]["headers"]
        assert headers["Authorization"] == "Bearer test_key_123"

    def test_timeout_is_passed(self, monkeypatch, mock_openrouter_post, fake_response):
        """Timeout of 120 should be passed to requests.post."""
        def fake_post(url, headers=None, json=None, timeout=None):
            mock_openrouter_post.append({"timeout": timeout})
            return fake_response(200, {"choices": [{"message": {"content": "ok"}}]})

        monkeypatch.setattr("requests.post", fake_post)

        openrouter_client.chat_completion([], "model", "key")

        assert mock_openrouter_post[0]["timeout"] == 120

    def test_messages_are_passed_to_payload(self, monkeypatch, mock_openrouter_post, fake_response):
        """Messages should be included in the payload exactly as provided."""
        def fake_post(url, headers=None, json=None, timeout=None):
            mock_openrouter_post.append({"json": json})
            return fake_response(200, {"choices": [{"message": {"content": "ok"}}]})

        monkeypatch.setattr("requests.post", fake_post)

        test_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        openrouter_client.chat_completion(test_messages, "model", "key")

        assert mock_openrouter_post[0]["json"]["messages"] == test_messages


# ============================================================================
# Tests: openrouter_client.caption_image
# ============================================================================


class TestCaptionImageDataURL:
    """Test caption_image data URL construction."""

    def test_caption_image_builds_data_url_png(self, monkeypatch, fake_response, tmp_path):
        """caption_image should build data:image/png;base64,... for .png files."""
        # Create a fake image file
        image_file = tmp_path / "test.png"
        image_data = b"fake_png_data"
        image_file.write_bytes(image_data)

        captured_json = []

        def fake_post(url, headers=None, json=None, timeout=None):
            captured_json.append(json)
            return fake_response(200, {"choices": [{"message": {"content": "ok"}}]})

        monkeypatch.setattr("requests.post", fake_post)

        openrouter_client.caption_image(image_file, "model", "key")

        messages = captured_json[0]["messages"]
        content = messages[0]["content"]
        image_url_obj = content[1]["image_url"]["url"]

        expected_b64 = base64.b64encode(image_data).decode("utf-8")
        assert image_url_obj == f"data:image/png;base64,{expected_b64}"

    def test_caption_image_builds_data_url_jpeg(self, monkeypatch, fake_response, tmp_path):
        """caption_image should build data:image/jpeg;base64,... for .jpeg files."""
        image_file = tmp_path / "test.jpeg"
        image_data = b"fake_jpeg_data"
        image_file.write_bytes(image_data)

        captured_json = []

        def fake_post(url, headers=None, json=None, timeout=None):
            captured_json.append(json)
            return fake_response(200, {"choices": [{"message": {"content": "ok"}}]})

        monkeypatch.setattr("requests.post", fake_post)

        openrouter_client.caption_image(image_file, "model", "key")

        messages = captured_json[0]["messages"]
        image_url_obj = messages[0]["content"][1]["image_url"]["url"]

        assert "data:image/jpeg;base64," in image_url_obj

    def test_caption_image_no_extension_defaults_to_png(self, monkeypatch, fake_response, tmp_path):
        """File with no extension should default to png."""
        image_file = tmp_path / "test_image"
        image_data = b"fake_image_data"
        image_file.write_bytes(image_data)

        captured_json = []

        def fake_post(url, headers=None, json=None, timeout=None):
            captured_json.append(json)
            return fake_response(200, {"choices": [{"message": {"content": "ok"}}]})

        monkeypatch.setattr("requests.post", fake_post)

        openrouter_client.caption_image(image_file, "model", "key")

        messages = captured_json[0]["messages"]
        image_url_obj = messages[0]["content"][1]["image_url"]["url"]

        assert "data:image/png;base64," in image_url_obj


class TestCaptionImagePrompt:
    """Test caption_image prompt construction."""

    def test_caption_image_prompt_includes_context_hint_first_500_chars(self, monkeypatch, fake_response, tmp_path):
        """Prompt should include first 500 chars of context_hint."""
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"data")

        captured_json = []

        def fake_post(url, headers=None, json=None, timeout=None):
            captured_json.append(json)
            return fake_response(200, {"choices": [{"message": {"content": "ok"}}]})

        monkeypatch.setattr("requests.post", fake_post)

        long_hint = "x" * 1000
        openrouter_client.caption_image(image_file, "model", "key", context_hint=long_hint)

        messages = captured_json[0]["messages"]
        prompt = messages[0]["content"][0]["text"]

        assert long_hint[:500] in prompt
        assert "Surrounding page text" in prompt

    def test_caption_image_no_hint_excludes_surrounding_text_section(self, monkeypatch, fake_response, tmp_path):
        """Prompt should not have 'Surrounding page text' when context_hint is empty."""
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"data")

        captured_json = []

        def fake_post(url, headers=None, json=None, timeout=None):
            captured_json.append(json)
            return fake_response(200, {"choices": [{"message": {"content": "ok"}}]})

        monkeypatch.setattr("requests.post", fake_post)

        openrouter_client.caption_image(image_file, "model", "key", context_hint="")

        messages = captured_json[0]["messages"]
        prompt = messages[0]["content"][0]["text"]

        assert "Surrounding page text" not in prompt

    def test_caption_image_max_tokens_300(self, monkeypatch, fake_response, tmp_path):
        """caption_image should pass max_tokens=300."""
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"data")

        captured_json = []

        def fake_post(url, headers=None, json=None, timeout=None):
            captured_json.append(json)
            return fake_response(200, {"choices": [{"message": {"content": "ok"}}]})

        monkeypatch.setattr("requests.post", fake_post)

        openrouter_client.caption_image(image_file, "model", "key")

        assert captured_json[0]["max_tokens"] == 300

    def test_caption_image_message_structure(self, monkeypatch, fake_response, tmp_path):
        """Message should have one text part and one image_url part."""
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"data")

        captured_json = []

        def fake_post(url, headers=None, json=None, timeout=None):
            captured_json.append(json)
            return fake_response(200, {"choices": [{"message": {"content": "ok"}}]})

        monkeypatch.setattr("requests.post", fake_post)

        openrouter_client.caption_image(image_file, "model", "key")

        messages = captured_json[0]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert "url" in content[1]["image_url"]


# ============================================================================
# Tests: RAGPipeline.answer
# ============================================================================


class TestRAGPipelineAnswer:
    """Test RAGPipeline.answer with mocked embedder and openrouter_client."""

    def test_answer_with_empty_store_returns_upload_message(self, monkeypatch, tmp_path):
        """answer() should return upload message when store is not ready."""
        # Use isolated pipeline with no index file
        monkeypatch.setattr(config, "INDEX_PATH", tmp_path / "nonexistent_index.npz")
        pipeline = RAGPipeline()
        # Don't populate the store

        text, results = pipeline.answer("test", "key", "model")

        assert "Upload a PDF" in text
        assert results == []

    def test_answer_formats_context_with_tags(self, monkeypatch, fake_embedder):
        """answer() should format context blocks with [TYPE | page N] tags."""
        pipeline = RAGPipeline()
        pipeline._embedder = fake_embedder

        # Populate store with fake data
        test_docs = [
            {
                "type": "text",
                "page": 1,
                "content": "Revenue was $100M",
                "dense_score": 0.95,
            },
            {
                "type": "table",
                "page": 2,
                "content": "| Q1 | Q2 |\n| 20 | 25 |",
                "dense_score": 0.90,
            },
        ]
        embeddings = np.random.randn(2, 384).astype(np.float32)
        pipeline.store.build(embeddings, test_docs)

        captured_messages = []

        def fake_chat(messages, model, api_key, **kwargs):
            captured_messages.extend(messages)
            return "Test answer"

        monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat)

        pipeline.answer("revenue", "key", "model")

        # Find the user message
        user_msg = [m for m in captured_messages if m["role"] == "user"][0]
        context = user_msg["content"]

        assert "[TEXT | page 1]" in context
        assert "[TABLE | page 2]" in context
        assert "Revenue was $100M" in context

    def test_answer_context_blocks_separated_by_delimiter(self, monkeypatch, fake_embedder):
        """Context blocks should be separated by '\\n\\n---\\n\\n'."""
        pipeline = RAGPipeline()
        pipeline._embedder = fake_embedder

        test_docs = [
            {"type": "text", "page": 1, "content": "Block 1", "dense_score": 0.95},
            {"type": "text", "page": 2, "content": "Block 2", "dense_score": 0.90},
            {"type": "text", "page": 3, "content": "Block 3", "dense_score": 0.85},
        ]
        embeddings = np.random.randn(3, 384).astype(np.float32)
        pipeline.store.build(embeddings, test_docs)

        captured_messages = []

        def fake_chat(messages, model, api_key, **kwargs):
            captured_messages.extend(messages)
            return "Test answer"

        monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat)

        pipeline.answer("test", "key", "model")

        user_msg = [m for m in captured_messages if m["role"] == "user"][0]
        context = user_msg["content"]

        # Count separators
        separator_count = context.count("\n\n---\n\n")
        assert separator_count == 2  # 3 blocks = 2 separators

    def test_answer_includes_question_in_prompt(self, monkeypatch, fake_embedder):
        """Question should appear in the user message."""
        pipeline = RAGPipeline()
        pipeline._embedder = fake_embedder

        test_docs = [
            {"type": "text", "page": 1, "content": "Some content", "dense_score": 0.95},
        ]
        embeddings = np.random.randn(1, 384).astype(np.float32)
        pipeline.store.build(embeddings, test_docs)

        captured_messages = []

        def fake_chat(messages, model, api_key, **kwargs):
            captured_messages.extend(messages)
            return "Test answer"

        monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat)

        question = "What was the revenue in Q3?"
        pipeline.answer(question, "key", "model")

        user_msg = [m for m in captured_messages if m["role"] == "user"][0]
        assert question in user_msg["content"]

    def test_answer_respects_top_k(self, monkeypatch, fake_embedder):
        """answer() should retrieve only top_k results."""
        pipeline = RAGPipeline()
        pipeline._embedder = fake_embedder

        # Create 10 documents
        test_docs = [
            {
                "type": "text",
                "page": i,
                "content": f"Content {i}",
                "dense_score": 1.0 - (i * 0.05),
            }
            for i in range(10)
        ]
        embeddings = np.random.randn(10, 384).astype(np.float32)
        pipeline.store.build(embeddings, test_docs)

        captured_results = []

        def fake_chat(messages, model, api_key, **kwargs):
            return "Test answer"

        monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat)

        text, results = pipeline.answer("test", "key", "model", top_k=3)

        assert len(results) == 3

    def test_answer_returns_text_and_results_tuple(self, monkeypatch, fake_embedder):
        """answer() should return (answer_text, results_list) tuple."""
        pipeline = RAGPipeline()
        pipeline._embedder = fake_embedder

        test_docs = [
            {"type": "text", "page": 1, "content": "Content", "dense_score": 0.95},
        ]
        embeddings = np.random.randn(1, 384).astype(np.float32)
        pipeline.store.build(embeddings, test_docs)

        def fake_chat(messages, model, api_key, **kwargs):
            return "Mock answer text"

        monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat)

        result = pipeline.answer("test", "key", "model")

        assert isinstance(result, tuple)
        assert len(result) == 2
        answer_text, results = result
        assert answer_text == "Mock answer text"
        assert isinstance(results, list)
        assert len(results) == 1

    def test_answer_passes_config_values_to_chat_completion(self, monkeypatch, fake_embedder):
        """answer() should pass TEXT_TEMPERATURE, TOP_P, TOP_K, MAX_TOKENS from config."""
        pipeline = RAGPipeline()
        pipeline._embedder = fake_embedder

        test_docs = [
            {"type": "text", "page": 1, "content": "Content", "dense_score": 0.95},
        ]
        embeddings = np.random.randn(1, 384).astype(np.float32)
        pipeline.store.build(embeddings, test_docs)

        captured_kwargs = {}

        def fake_chat(messages, model, api_key, **kwargs):
            captured_kwargs.update(kwargs)
            return "ok"

        monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat)

        pipeline.answer("test", "key", "model")

        assert captured_kwargs["temperature"] == config.TEXT_TEMPERATURE
        assert captured_kwargs["top_p"] == config.TEXT_TOP_P
        assert captured_kwargs["top_k"] == config.TEXT_TOP_K
        assert captured_kwargs["max_tokens"] == config.TEXT_MAX_TOKENS


# ============================================================================
# Tests: app.py helper functions
# ============================================================================


class TestFormatSources:
    """Test app.format_sources function."""

    def test_format_sources_empty_results_returns_default(self):
        """format_sources should return default message for empty results."""
        result = app.format_sources([])
        assert "No sources retrieved" in result

    def test_format_sources_includes_type_emoji(self):
        """format_sources should include emoji for type."""
        results = [
            {
                "type": "text",
                "page": 1,
                "content": "Some text",
                "dense_score": 0.95,
            }
        ]
        result = app.format_sources(results)
        assert "📄" in result

    def test_format_sources_includes_page_number(self):
        """format_sources should include page number."""
        results = [
            {
                "type": "text",
                "page": 5,
                "content": "Some text",
                "dense_score": 0.95,
            }
        ]
        result = app.format_sources(results)
        assert "page 5" in result

    def test_format_sources_includes_dense_score(self):
        """format_sources should include dense_score formatted to 2 decimals."""
        results = [
            {
                "type": "text",
                "page": 1,
                "content": "Some text",
                "dense_score": 0.9513,
            }
        ]
        result = app.format_sources(results)
        assert "0.95" in result

    def test_format_sources_truncates_long_snippet(self):
        """Snippet longer than 220 chars should be truncated with ellipsis."""
        long_content = "x" * 250
        results = [
            {
                "type": "text",
                "page": 1,
                "content": long_content,
                "dense_score": 0.95,
            }
        ]
        result = app.format_sources(results)
        assert "..." in result
        # Snippet should be truncated to ~220 chars + "..."
        assert "x" * 220 in result
        assert "x" * 230 not in result

    def test_format_sources_multiple_results_separated_by_separator(self):
        """Multiple results should be separated by '\\n\\n---\\n\\n'."""
        results = [
            {"type": "text", "page": 1, "content": "First", "dense_score": 0.95},
            {"type": "table", "page": 2, "content": "Second", "dense_score": 0.90},
        ]
        result = app.format_sources(results)
        assert result.count("\n\n---\n\n") == 1

    def test_format_sources_newlines_replaced_with_spaces(self):
        """Newlines in content should be replaced with spaces."""
        results = [
            {
                "type": "text",
                "page": 1,
                "content": "Line 1\nLine 2\nLine 3",
                "dense_score": 0.95,
            }
        ]
        result = app.format_sources(results)
        assert "\nLine 2" not in result
        assert "Line 1 Line 2 Line 3" in result


class TestCollectFigurePaths:
    """Test app.collect_figure_paths function."""

    def test_collect_figure_paths_extracts_image_paths(self):
        """collect_figure_paths should extract image_path for type='figure'."""
        results = [
            {"type": "figure", "image_path": "/path/to/fig1.png", "page": 1},
            {"type": "figure", "image_path": "/path/to/fig2.png", "page": 2},
        ]
        paths = app.collect_figure_paths(results)
        assert paths == ["/path/to/fig1.png", "/path/to/fig2.png"]

    def test_collect_figure_paths_ignores_non_figure_types(self):
        """collect_figure_paths should only include type='figure'."""
        results = [
            {"type": "text", "image_path": "/ignored", "page": 1},
            {"type": "figure", "image_path": "/path/to/fig.png", "page": 2},
            {"type": "table", "image_path": "/also_ignored", "page": 3},
        ]
        paths = app.collect_figure_paths(results)
        assert len(paths) == 1
        assert "/path/to/fig.png" in paths

    def test_collect_figure_paths_ignores_missing_image_path(self):
        """Figures without image_path should be skipped."""
        results = [
            {"type": "figure", "page": 1},  # No image_path
            {"type": "figure", "image_path": "/path/to/fig.png", "page": 2},
        ]
        paths = app.collect_figure_paths(results)
        assert len(paths) == 1
        assert "/path/to/fig.png" in paths

    def test_collect_figure_paths_empty_results(self):
        """collect_figure_paths should return empty list for empty results."""
        paths = app.collect_figure_paths([])
        assert paths == []

    def test_collect_figure_paths_preserves_order(self):
        """collect_figure_paths should preserve order of results."""
        results = [
            {"type": "figure", "image_path": "/fig1.png", "page": 1},
            {"type": "figure", "image_path": "/fig2.png", "page": 2},
            {"type": "figure", "image_path": "/fig3.png", "page": 3},
        ]
        paths = app.collect_figure_paths(results)
        assert paths == ["/fig1.png", "/fig2.png", "/fig3.png"]
