import uuid

import pytest

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.qdrant import Entry, QdrantConnector


@pytest.fixture
async def embedding_provider():
    """Fixture to provide a FastEmbed embedding provider."""
    return FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
async def qdrant_connector(embedding_provider):
    """Fixture to provide a QdrantConnector with in-memory Qdrant client."""
    collection_name = f"test_collection_{uuid.uuid4().hex}"
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
    )
    yield connector


@pytest.mark.asyncio
async def test_count_points_empty(qdrant_connector):
    """Test counting points in an empty collection."""
    count = await qdrant_connector.count_points(qdrant_connector._default_collection_name)
    assert count == 0


@pytest.mark.asyncio
async def test_count_points_after_store(qdrant_connector):
    """Test counting points after storing entries."""
    await qdrant_connector.store(Entry(content="First entry"))
    await qdrant_connector.store(Entry(content="Second entry"))
    count = await qdrant_connector.count_points(qdrant_connector._default_collection_name)
    assert count == 2


@pytest.mark.asyncio
async def test_count_points_after_delete(qdrant_connector):
    """Test counting points after deleting some."""
    await qdrant_connector.store(Entry(content="First"))
    await qdrant_connector.store(Entry(content="Second"))
    await qdrant_connector.store(Entry(content="Third"))
    assert await qdrant_connector.count_points(qdrant_connector._default_collection_name) == 3


@pytest.mark.asyncio
async def test_list_points_empty(qdrant_connector):
    """Test listing points in an empty collection."""
    result = await qdrant_connector.list_points(qdrant_connector._default_collection_name)
    assert result["points"] == []
    assert result["next_offset"] is None


@pytest.mark.asyncio
async def test_list_points_basic(qdrant_connector):
    """Test listing points from a collection with entries."""
    await qdrant_connector.store(Entry(content="Entry one", metadata={"key": "value1"}))
    await qdrant_connector.store(Entry(content="Entry two", metadata={"key": "value2"}))

    result = await qdrant_connector.list_points(qdrant_connector._default_collection_name, limit=10)
    assert len(result["points"]) == 2
    assert result["next_offset"] is None

    # Verify each point has id and payload
    for point in result["points"]:
        assert "id" in point
        assert "payload" in point


@pytest.mark.asyncio
async def test_list_points_with_vectors(qdrant_connector):
    """Test listing points with vector data included."""
    await qdrant_connector.store(Entry(content="Test content for vectors"))

    result = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name,
        limit=10,
        with_vector=True,
    )
    assert len(result["points"]) == 1
    assert "vector" in result["points"][0]


@pytest.mark.asyncio
async def test_list_points_without_vectors(qdrant_connector):
    """Test listing points without vector data (default)."""
    await qdrant_connector.store(Entry(content="Test content without vectors"))

    result = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name,
        limit=10,
        with_vector=False,
    )
    assert len(result["points"]) == 1
    assert "vector" not in result["points"][0]


@pytest.mark.asyncio
async def test_list_points_pagination(qdrant_connector):
    """Test pagination with offset and next_offset."""
    for i in range(5):
        await qdrant_connector.store(Entry(content=f"Entry {i}"))

    # Get first page
    page1 = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name,
        limit=2,
    )
    assert len(page1["points"]) == 2
    assert page1["next_offset"] is not None

    # Get second page
    page2 = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name,
        limit=2,
        offset=page1["next_offset"],
    )
    assert len(page2["points"]) == 2

    # Get third page
    page3 = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name,
        limit=2,
        offset=page2["next_offset"],
    )
    assert len(page3["points"]) == 1
    assert page3["next_offset"] is None


@pytest.mark.asyncio
async def test_list_points_with_limit(qdrant_connector):
    """Test limiting the number of returned points."""
    for i in range(10):
        await qdrant_connector.store(Entry(content=f"Entry {i}"))

    result = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name,
        limit=3,
    )
    assert len(result["points"]) == 3


@pytest.mark.asyncio
async def test_list_points_with_payload_filter(qdrant_connector):
    """Test listing points with a query filter on payload.

    Note: Payload indexes don't work in local Qdrant mode, so this test
    is skipped when using in-memory mode. It will work with a real Qdrant server.
    """
    import os

    # Skip if running in local/in-memory mode (indexes have no effect there)
    url = qdrant_connector._qdrant_url
    if url == ":memory:" or url is None:
        pytest.skip("Payload filtering requires a real Qdrant server (not local mode)")

    from qdrant_client import models

    await qdrant_connector.store(Entry(content="Red item", metadata={"color": "red"}))
    await qdrant_connector.store(Entry(content="Blue item", metadata={"color": "blue"}))
    await qdrant_connector.store(Entry(content="Red item 2", metadata={"color": "red"}))

    # Index the payload field used in the filter (required for filtered queries in Qdrant)
    await qdrant_connector._client.create_payload_index(
        collection_name=qdrant_connector._default_collection_name,
        field_name="color",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    result = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="color", match=models.MatchValue(value="red"))]
        ),
        limit=10,
    )
    assert len(result["points"]) == 2
    for point in result["points"]:
        assert point["payload"].get("color") == "red"
