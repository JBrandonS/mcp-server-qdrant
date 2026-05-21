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
async def test_recommend_positive_only(qdrant_connector):
    """Test recommendation with positive examples only."""
    # Store similar entries
    await qdrant_connector.store(Entry(content="Machine learning algorithms"))
    await qdrant_connector.store(Entry(content="Deep neural networks"))
    await qdrant_connector.store(Entry(content="Python programming language"))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    positive_id = points["points"][0]["id"]

    result = await qdrant_connector.recommend(
        qdrant_connector._default_collection_name,
        positive=[positive_id],
        limit=5,
    )
    assert len(result) > 0
    # Result should include score
    assert "score" in result[0]


@pytest.mark.asyncio
async def test_recommend_with_limit(qdrant_connector):
    """Test recommendation respects limit parameter."""
    for i in range(5):
        await qdrant_connector.store(Entry(content=f"Document {i} about technology"))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    positive_id = points["points"][0]["id"]

    result = await qdrant_connector.recommend(
        qdrant_connector._default_collection_name,
        positive=[positive_id],
        limit=2,
    )
    assert len(result) <= 2


@pytest.mark.asyncio
async def test_recommend_with_vectors(qdrant_connector):
    """Test recommendation with vector data included."""
    # Need at least 2 points for recommendations to find similar ones
    await qdrant_connector.store(Entry(content="Tech content A"))
    await qdrant_connector.store(Entry(content="Tech content B - related to A"))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    positive_id = points["points"][0]["id"]

    result = await qdrant_connector.recommend(
        qdrant_connector._default_collection_name,
        positive=[positive_id],
        with_vector=True,
    )
    assert len(result) > 0
    assert "vector" in result[0]


@pytest.mark.asyncio
async def test_recommend_with_payload(qdrant_connector):
    """Test recommendation includes payload data."""
    # Need at least 2 points for recommendations to find similar ones
    await qdrant_connector.store(Entry(content="Content A", metadata={"source": "test"}))
    await qdrant_connector.store(Entry(content="Content B - related", metadata={"source": "test"}))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    positive_id = points["points"][0]["id"]

    result = await qdrant_connector.recommend(
        qdrant_connector._default_collection_name,
        positive=[positive_id],
        with_payload=True,
    )
    assert len(result) > 0
    assert "payload" in result[0]


@pytest.mark.asyncio
async def test_recommend_empty_result(qdrant_connector):
    """Test recommendation on empty collection returns empty list."""
    # Collection exists but has no points
    result = await qdrant_connector.recommend(
        qdrant_connector._default_collection_name,
        positive=["nonexistent"],
    )
    assert result == []
