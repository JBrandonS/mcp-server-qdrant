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
async def test_get_point_single(qdrant_connector):
    """Test retrieving a single point by ID."""
    await qdrant_connector.store(Entry(content="Single entry", metadata={"key": "val"}))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    point_id = points["points"][0]["id"]

    result = await qdrant_connector.get_point(
        qdrant_connector._default_collection_name, [point_id]
    )
    assert len(result) == 1
    assert result[0]["id"] == point_id
    assert "payload" in result[0]


@pytest.mark.asyncio
async def test_get_point_multiple(qdrant_connector):
    """Test retrieving multiple points by IDs."""
    await qdrant_connector.store(Entry(content="First"))
    await qdrant_connector.store(Entry(content="Second"))
    await qdrant_connector.store(Entry(content="Third"))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    ids = [p["id"] for p in points["points"][:2]]

    result = await qdrant_connector.get_point(
        qdrant_connector._default_collection_name, ids
    )
    assert len(result) == 2
    result_ids = {r["id"] for r in result}
    assert result_ids == set(ids)


@pytest.mark.asyncio
async def test_get_point_with_vectors(qdrant_connector):
    """Test retrieving points with vector data."""
    await qdrant_connector.store(Entry(content="Content"))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    point_id = points["points"][0]["id"]

    result = await qdrant_connector.get_point(
        qdrant_connector._default_collection_name, [point_id], with_vector=True
    )
    assert len(result) == 1
    assert "vector" in result[0]


@pytest.mark.asyncio
async def test_get_point_without_payload(qdrant_connector):
    """Test retrieving points without payload data."""
    await qdrant_connector.store(Entry(content="Content"))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    point_id = points["points"][0]["id"]

    result = await qdrant_connector.get_point(
        qdrant_connector._default_collection_name, [point_id], with_payload=False
    )
    assert len(result) == 1
    assert "payload" not in result[0] or result[0].get("payload") is None


@pytest.mark.asyncio
async def test_get_point_nonexistent(qdrant_connector):
    """Test retrieving a non-existent point returns empty list."""
    result = await qdrant_connector.get_point(
        qdrant_connector._default_collection_name, ["nonexistent_id"]
    )
    assert result == []
