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
async def test_update_points_single(qdrant_connector):
    """Test updating payload on a single point."""
    await qdrant_connector.store(Entry(content="Original", metadata={"key": "old"}))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    point_id = points["points"][0]["id"]

    result = await qdrant_connector.update_points(
        qdrant_connector._default_collection_name,
        [{"id": point_id, "payload": {"key": "new"}}],
    )
    assert "Updated payloads" in result

    # Verify the update
    updated = await qdrant_connector.get_point(
        qdrant_connector._default_collection_name, [point_id]
    )
    assert updated[0]["payload"]["key"] == "new"


@pytest.mark.asyncio
async def test_update_points_multiple(qdrant_connector):
    """Test updating payloads on multiple points."""
    await qdrant_connector.store(Entry(content="First", metadata={"index": 0}))
    await qdrant_connector.store(Entry(content="Second", metadata={"index": 1}))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    ids = [p["id"] for p in points["points"]]

    result = await qdrant_connector.update_points(
        qdrant_connector._default_collection_name,
        [
            {"id": ids[0], "payload": {"updated": True}},
            {"id": ids[1], "payload": {"updated": True}},
        ],
    )
    assert str(len(ids)) in result

    # Verify both updated
    updated = await qdrant_connector.get_point(
        qdrant_connector._default_collection_name, ids
    )
    for point in updated:
        assert point["payload"].get("updated") is True


@pytest.mark.asyncio
async def test_update_points_without_payload(qdrant_connector):
    """Test updating point with just an ID (no payload change)."""
    await qdrant_connector.store(Entry(content="Content", metadata={"key": "val"}))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    point_id = points["points"][0]["id"]

    result = await qdrant_connector.update_points(
        qdrant_connector._default_collection_name,
        [{"id": point_id}],
    )
    assert "Updated payloads" in result


@pytest.mark.asyncio
async def test_update_points_verify_preserves_other_keys(qdrant_connector):
    """Test that updating payload merges rather than overwrites entirely."""
    await qdrant_connector.store(Entry(content="Content", metadata={"a": 1, "b": 2}))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10
    )
    point_id = points["points"][0]["id"]

    # Update with only key 'c'
    await qdrant_connector.update_points(
        qdrant_connector._default_collection_name,
        [{"id": point_id, "payload": {"c": 3}}],
    )

    updated = await qdrant_connector.get_point(
        qdrant_connector._default_collection_name, [point_id]
    )
    # Qdrant's set_payload replaces the entire payload for each point
    assert updated[0]["payload"]["c"] == 3


@pytest.mark.asyncio
async def test_update_vectors_single(qdrant_connector):
    """Test updating vector on a single point."""
    await qdrant_connector.store(Entry(content="Content"))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10, with_vector=True
    )
    original_vector_data = points["points"][0]["vector"]
    point_id = points["points"][0]["id"]

    # Handle named vectors: vector may be {name: [values]} or [values]
    if isinstance(original_vector_data, dict):
        vector_name = next(iter(original_vector_data.keys()))
        original_dim = len(original_vector_data[vector_name])
    else:
        original_dim = len(original_vector_data)

    # Create a new vector (same dimension as the embedding model)
    new_vector = [0.0] * original_dim

    result = await qdrant_connector.update_vectors(
        qdrant_connector._default_collection_name,
        [{"id": point_id, "vector": new_vector}],
    )
    assert "Updated vectors" in result

    # Verify the update
    updated = await qdrant_connector.get_point(
        qdrant_connector._default_collection_name, [point_id], with_vector=True
    )
    updated_vector_data = updated[0]["vector"]
    if isinstance(updated_vector_data, dict):
        assert list(updated_vector_data.values())[0] == new_vector
    else:
        assert updated_vector_data == new_vector


@pytest.mark.asyncio
async def test_update_vectors_multiple(qdrant_connector):
    """Test updating vectors on multiple points."""
    await qdrant_connector.store(Entry(content="First"))
    await qdrant_connector.store(Entry(content="Second"))

    points = await qdrant_connector.list_points(
        qdrant_connector._default_collection_name, limit=10, with_vector=True
    )
    ids = [p["id"] for p in points["points"]]

    # Determine vector dimension from first point
    vec_data = points["points"][0]["vector"]
    dim = len(vec_data[next(iter(vec_data.keys()))]) if isinstance(vec_data, dict) else len(vec_data)

    # Use unit vectors with different directions (Cosine normalizes, so use orthogonal-ish patterns)
    new_vectors = [[1.0] + [0.0] * (dim - 1), [0.0] * (dim - 1) + [1.0]]

    result = await qdrant_connector.update_vectors(
        qdrant_connector._default_collection_name,
        [{"id": ids[0], "vector": new_vectors[0]}, {"id": ids[1], "vector": new_vectors[1]}],
    )
    assert str(len(ids)) in result

    # Verify both updated with distinct vectors
    for idx in range(len(ids)):
        updated = await qdrant_connector.get_point(
            qdrant_connector._default_collection_name, [ids[idx]], with_vector=True
        )
        vec = updated[0]["vector"]
        expected = new_vectors[idx]
        if isinstance(vec, dict):
            actual = list(vec.values())[0]
        else:
            actual = vec
        # Use near-equal comparison for floating point
        assert all(abs(a - e) < 0.001 for a, e in zip(actual, expected)), \
            f"Point {ids[idx]} vector mismatch: got {actual[:5]}..., expected {expected[:5]}"
