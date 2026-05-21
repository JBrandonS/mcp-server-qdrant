import uuid

import pytest
from qdrant_client import models

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.qdrant import QdrantConnector


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
async def test_list_collections_empty(qdrant_connector):
    """Test listing collections when none exist."""
    collections = await qdrant_connector.list_collections()
    assert collections == []


@pytest.mark.asyncio
async def test_list_collections_single(qdrant_connector):
    """Test listing collections with one collection."""
    name = f"single_collection_{uuid.uuid4().hex}"
    await qdrant_connector.create_collection(name, vector_size=384)
    collections = await qdrant_connector.list_collections()
    assert name in collections


@pytest.mark.asyncio
async def test_list_collections_multiple(qdrant_connector):
    """Test listing collections with multiple collections."""
    names = [f"coll_{uuid.uuid4().hex[:8]}" for _ in range(3)]
    for name in names:
        await qdrant_connector.create_collection(name, vector_size=384)
    collections = await qdrant_connector.list_collections()
    for name in names:
        assert name in collections


@pytest.mark.asyncio
async def test_get_collection_info_existing(qdrant_connector):
    """Test getting info for an existing collection."""
    name = f"info_collection_{uuid.uuid4().hex}"
    await qdrant_connector.create_collection(name, vector_size=384)
    info = await qdrant_connector.get_collection_info(name)
    assert "status" in info
    assert "indexed_vectors_count" in info
    assert "points_count" in info
    assert "segments_count" in info
    assert "config" in info
    assert "params" in info["config"]
    assert "hnsw_config" in info["config"]
    assert "optimizers_config" in info["config"]


@pytest.mark.asyncio
async def test_get_collection_info_nonexistent(qdrant_connector):
    """Test getting info for a non-existent collection raises an error."""
    with pytest.raises(Exception):
        await qdrant_connector.get_collection_info(f"nonexistent_{uuid.uuid4().hex}")


@pytest.mark.asyncio
async def test_create_collection_default_distance(qdrant_connector):
    """Test creating a collection with default cosine distance."""
    name = f"create_default_{uuid.uuid4().hex}"
    result = await qdrant_connector.create_collection(name, vector_size=384)
    assert "created successfully" in result
    collections = await qdrant_connector.list_collections()
    assert name in collections


@pytest.mark.asyncio
async def test_create_collection_custom_distance(qdrant_connector):
    """Test creating a collection with a custom distance metric."""
    name = f"create_euclid_{uuid.uuid4().hex}"
    result = await qdrant_connector.create_collection(
        name, vector_size=384, distance="Euclid"
    )
    assert "created successfully" in result
    info = await qdrant_connector.get_collection_info(name)
    assert info["config"]["params"]["vectors"]["distance"] == "Euclid"


@pytest.mark.asyncio
async def test_create_collection_duplicate_raises(qdrant_connector):
    """Test that creating a duplicate collection raises an error."""
    name = f"create_dup_{uuid.uuid4().hex}"
    await qdrant_connector.create_collection(name, vector_size=384)
    with pytest.raises(Exception):
        await qdrant_connector.create_collection(name, vector_size=384)


@pytest.mark.asyncio
async def test_update_collection_optimizer(qdrant_connector):
    """Test updating collection optimizer config."""
    name = f"update_optim_{uuid.uuid4().hex}"
    await qdrant_connector.create_collection(name, vector_size=384)
    result = await qdrant_connector.update_collection(
        name, optimizer_config={"indexing_threshold": 10000}
    )
    assert "updated successfully" in result


@pytest.mark.asyncio
async def test_update_collection_replication_factor(qdrant_connector):
    """Test updating collection replication factor."""
    name = f"update_repl_{uuid.uuid4().hex}"
    await qdrant_connector.create_collection(name, vector_size=384)
    result = await qdrant_connector.update_collection(
        name, replication_factor=2
    )
    assert "updated successfully" in result


@pytest.mark.asyncio
async def test_update_collection_write_consistency_factor(qdrant_connector):
    """Test updating collection write consistency factor."""
    name = f"update_write_{uuid.uuid4().hex}"
    await qdrant_connector.create_collection(name, vector_size=384)
    result = await qdrant_connector.update_collection(
        name, write_consistency_factor=2
    )
    assert "updated successfully" in result


@pytest.mark.asyncio
async def test_delete_collection_success(qdrant_connector):
    """Test deleting an existing collection."""
    name = f"delete_coll_{uuid.uuid4().hex}"
    await qdrant_connector.create_collection(name, vector_size=384)
    result = await qdrant_connector.delete_collection(name)
    assert "deleted successfully" in result
    collections = await qdrant_connector.list_collections()
    assert name not in collections


@pytest.mark.asyncio
async def test_delete_collection_nonexistent(qdrant_connector):
    """Test deleting a non-existent collection."""
    # In :memory: mode, this may or may not raise; we just verify it doesn't crash the connector
    result = await qdrant_connector.delete_collection(f"nonexistent_{uuid.uuid4().hex}")
    assert "deleted successfully" in result


@pytest.mark.asyncio
async def test_delete_collection_verify_removed(qdrant_connector):
    """Test that deleted collection no longer appears in list."""
    name = f"delete_verify_{uuid.uuid4().hex}"
    await qdrant_connector.create_collection(name, vector_size=384)
    await qdrant_connector.delete_collection(name)
    assert await qdrant_connector._client.collection_exists(name) is False
