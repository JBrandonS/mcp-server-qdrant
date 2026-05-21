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
async def test_delete_points_single(qdrant_connector):
    """Test deleting a single point."""
    await qdrant_connector.store(Entry(content="To delete"))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    point_id = points["points"][0]["id"]

    result = await qdrant_connector.delete_points(
        qdrant_connector._default_collection_name, [point_id]
    )
    assert "Deleted 1 points" in result

    # Verify deletion
    remaining = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    assert remaining["points"] == []


@pytest.mark.asyncio
async def test_delete_points_multiple(qdrant_connector):
    """Test deleting multiple points."""
    await qdrant_connector.store(Entry(content="First"))
    await qdrant_connector.store(Entry(content="Second"))
    await qdrant_connector.store(Entry(content="Third"))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    ids_to_delete = [p["id"] for p in points["points"][:2]]

    result = await qdrant_connector.delete_points(
        qdrant_connector._default_collection_name, ids_to_delete
    )
    assert "Deleted 2 points" in result

    remaining = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    assert len(remaining["points"]) == 1


@pytest.mark.asyncio
async def test_delete_points_verify_count(qdrant_connector):
    """Test that count reflects deleted points."""
    await qdrant_connector.store(Entry(content="A"))
    await qdrant_connector.store(Entry(content="B"))

    assert await qdrant_connector.count_points(
        qdrant_connector._default_collection_name
    ) == 2

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    await qdrant_connector.delete_points(
        qdrant_connector._default_collection_name, [p["id"] for p in points["points"]]
    )

    assert await qdrant_connector.count_points(
        qdrant_connector._default_collection_name
    ) == 0


@pytest.mark.asyncio
async def test_delete_vectors(qdrant_connector):
    """Test deleting vectors from points."""
    await qdrant_connector.store(Entry(content="Content"))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10, with_vector=True
    )
    point_id = points["points"][0]["id"]

    result = await qdrant_connector.delete_vectors(
        qdrant_connector._default_collection_name, [point_id]
    )
    assert "from 1 points" in result

    # Verify vectors are gone but point still exists
    retrieved = await qdrant_connector.get_point(
        qdrant_connector._default_collection_name, [point_id], with_vector=True
    )
    assert len(retrieved) == 1
    assert "vector" not in retrieved[0] or retrieved[0].get("vector") is None


@pytest.mark.asyncio
async def test_delete_vectors_specific_names(qdrant_connector):
    """Test deleting specific named vectors."""
    await qdrant_connector.store(Entry(content="Content"))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10, with_vector=True
    )
    point_id = points["points"][0]["id"]

    # Get vector name from the stored point
    vector_name = list(points["points"][0]["vector"].keys())[0]

    result = await qdrant_connector.delete_vectors(
        qdrant_connector._default_collection_name, [point_id], vector_names=[vector_name]
    )
    assert vector_name in result
