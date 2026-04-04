"""Tests for SpatialMemoryService — Backboard assistant/thread/message lifecycle.

All tests mock BackboardClient to avoid real API calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_backboard_client():
    """Create a fully mocked BackboardClient."""
    client = AsyncMock()

    # create_assistant returns object with assistant_id
    assistant_obj = MagicMock()
    assistant_obj.assistant_id = "ast_123"
    client.create_assistant = AsyncMock(return_value=assistant_obj)

    # create_thread returns objects with unique thread_ids (cycling)
    thread_counter = {"n": 0}
    thread_ids = ["thr_scene1", "thr_scene2", "thr_scene3"]

    async def _create_thread(assistant_id):
        idx = thread_counter["n"]
        thread_counter["n"] += 1
        obj = MagicMock()
        obj.thread_id = thread_ids[idx % len(thread_ids)]
        return obj

    client.create_thread = AsyncMock(side_effect=_create_thread)

    # add_message returns object with content
    response_obj = MagicMock()
    response_obj.content = "LLM response text"
    client.add_message = AsyncMock(return_value=response_obj)

    return client


@pytest.fixture
def memory_service(mock_backboard_client):
    """Create a SpatialMemoryService with mocked client."""
    with patch(
        "server.services.backboard_client.BackboardClient",
        return_value=mock_backboard_client,
    ), patch(
        "server.services.backboard_client.BACKBOARD_AVAILABLE",
        True,
    ):
        from server.services.backboard_client import SpatialMemoryService

        service = SpatialMemoryService(api_key="test_key")
        # Replace the client with our mock directly
        service.client = mock_backboard_client
        return service


@pytest.mark.asyncio
async def test_initialize_creates_assistant_with_spatial_prompt(memory_service):
    """Test 1: initialize() creates assistant with name='SpatialMind' and spatial system prompt."""
    await memory_service.initialize()

    memory_service.client.create_assistant.assert_awaited_once()
    call_kwargs = memory_service.client.create_assistant.call_args
    assert call_kwargs.kwargs.get("name") == "SpatialMind" or (
        call_kwargs.args and call_kwargs.args[0] == "SpatialMind"
    )
    # Check system_prompt contains spatial reasoning keywords
    prompt_arg = call_kwargs.kwargs.get("system_prompt", "")
    assert "spatial" in prompt_arg.lower()
    assert memory_service.assistant_id == "ast_123"


@pytest.mark.asyncio
async def test_get_or_create_thread_creates_new_then_returns_cached(
    memory_service,
):
    """Test 2: get_or_create_thread creates new thread on first call, returns same on second."""
    await memory_service.initialize()

    thread_id_1 = await memory_service.get_or_create_thread("scene_a")
    thread_id_2 = await memory_service.get_or_create_thread("scene_a")

    assert thread_id_1 == thread_id_2
    # create_thread should only be called once for the same scene
    assert memory_service.client.create_thread.await_count == 1


@pytest.mark.asyncio
async def test_different_scenes_get_different_threads(memory_service):
    """Test 3: Different scene_ids get different thread_ids (BB-02)."""
    await memory_service.initialize()

    thread_a = await memory_service.get_or_create_thread("scene_a")
    thread_b = await memory_service.get_or_create_thread("scene_b")

    assert thread_a != thread_b
    assert memory_service.client.create_thread.await_count == 2


@pytest.mark.asyncio
async def test_store_catalog_calls_add_message(memory_service):
    """Test 4: store_catalog sends catalog content via add_message."""
    await memory_service.initialize()
    catalog_text = "Scene Object Catalog:\n- table at (1.0, 0.5, 2.0)"

    await memory_service.store_catalog("scene_a", catalog_text)

    memory_service.client.add_message.assert_awaited_once()
    call_kwargs = memory_service.client.add_message.call_args
    assert call_kwargs.kwargs["content"] == catalog_text


@pytest.mark.asyncio
async def test_query_with_context_returns_response(memory_service):
    """Test 5: query_with_context sends enriched message and returns response content."""
    await memory_service.initialize()

    result = await memory_service.query_with_context(
        scene_id="scene_a",
        query="What is on the table?",
        spatial_context="table is next_to chair",
    )

    assert result == "LLM response text"
    call_kwargs = memory_service.client.add_message.call_args
    content_arg = str(call_kwargs)
    assert "Spatial context" in content_arg
    assert "What is on the table?" in content_arg


@pytest.mark.asyncio
async def test_cross_session_recall_persists_thread_ids(memory_service):
    """Test 6: scene_threads dict persists thread_ids for cross-session recall (BB-06)."""
    await memory_service.initialize()

    # First session: create thread for scene
    thread_id = await memory_service.get_or_create_thread("scene_a")
    assert "scene_a" in memory_service.scene_threads
    assert memory_service.scene_threads["scene_a"] == thread_id

    # Simulate revisit: same service instance, same scene
    thread_id_revisit = await memory_service.get_or_create_thread("scene_a")
    assert thread_id_revisit == thread_id
    # Only one create_thread call — the second was served from cache
    assert memory_service.client.create_thread.await_count == 1
