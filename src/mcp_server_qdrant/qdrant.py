import logging
import uuid
from typing import Any

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.settings import METADATA_PATH

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        self._field_indexes = field_indexes

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def list_collections(self) -> list[str]:
        """
        List all collections in the Qdrant server.
        :return: A list of collection name strings.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def get_collection_info(self, collection_name: str) -> dict:
        """
        Get detailed information about a collection.
        :param collection_name: The name of the collection.
        :return: A dict with stats and config.
        """
        info = await self._client.get_collection(collection_name)
        return {
            "status": info.status.value,
            "indexed_vectors_count": info.indexed_vectors_count or 0,
            "points_count": info.points_count,
            "segments_count": info.segments_count,
            "config": {
                "params": info.config.params.model_dump(mode="json"),
                "hnsw_config": info.config.hnsw_config.model_dump(mode="json"),
                "optimizers_config": info.config.optimizer_config.model_dump(mode="json"),
            },
        }

    _DISTANCE_MAP = {
        "Cosine": models.Distance.COSINE,
        "Euclid": models.Distance.EUCLID,
        "Dot": models.Distance.DOT,
        "Manhattan": models.Distance.MANHATTAN,
    }

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> str:
        """
        Create a new Qdrant collection with vector configuration.
        :param collection_name: The name for the new collection.
        :param vector_size: The vector dimension matching your embedding model.
        :param distance: The distance metric to use (default: "Cosine").
        :return: A confirmation message.
        """
        distance_enum = self._DISTANCE_MAP.get(distance, models.Distance.COSINE)
        await self._client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=distance_enum
            ),
        )
        if self._field_indexes:
            for field_name, field_type in self._field_indexes.items():
                await self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
        return f"Collection '{collection_name}' created successfully"

    async def update_collection(
        self,
        collection_name: str,
        optimizer_config: dict | None = None,
        replication_factor: int | None = None,
        write_consistency_factor: int | None = None,
    ) -> str:
        """
        Update Qdrant collection settings.
        :param collection_name: The name of the collection.
        :param optimizer_config: Optional dict with optimizer settings.
        :param replication_factor: Optional number of shard copies.
        :param write_consistency_factor: Optional how many replicas must ack.
        :return: A confirmation message.
        """
        kwargs: dict[str, Any] = {}
        if optimizer_config is not None:
            kwargs["optimizers_config"] = models.OptimizersConfigDiff(**optimizer_config)
        if replication_factor is not None or write_consistency_factor is not None:
            params_diff = models.CollectionParamsDiff(
                replication_factor=replication_factor,
                write_consistency_factor=write_consistency_factor,
            )
            kwargs["collection_params"] = params_diff
        await self._client.update_collection(collection_name, **kwargs)
        return f"Collection '{collection_name}' updated successfully"

    async def delete_collection(self, collection_name: str) -> str:
        """
        Delete a Qdrant collection and all its data.
        :param collection_name: The name of the collection to delete.
        :return: A confirmation message.
        """
        await self._client.delete_collection(collection_name=collection_name)
        return f"Collection '{collection_name}' deleted successfully"

    async def list_points(
        self,
        collection_name: str,
        query_filter: models.Filter | None = None,
        limit: int = 10,
        offset: models.ExtendedPointId | None = None,
        with_vector: bool | list[str] = False,
        with_payload: bool | list[str] = True,
    ) -> dict:
        """
        List points from a Qdrant collection with optional filtering and pagination.
        :param collection_name: The name of the collection.
        :param query_filter: Optional filter to apply.
        :param limit: Maximum number of points to return.
        :param offset: Offset for pagination.
        :param with_vector: Whether to include vector data.
        :param with_payload: Whether to include payload data.
        :return: Dict with 'points' list and 'next_offset'.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return {"points": [], "next_offset": None}

        points, next_offset = await self._client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vector,
        )
        return {
            "points": [
                {
                    "id": point.id,
                    "payload": point.payload or {},
                    **({"vector": point.vector} if point.vector else {}),
                }
                for point in points
            ],
            "next_offset": next_offset,
        }

    async def count_points(
        self,
        collection_name: str,
        query_filter: models.Filter | None = None,
    ) -> int:
        """
        Count points in a Qdrant collection, optionally filtered.
        :param collection_name: The name of the collection.
        :param query_filter: Optional filter to apply.
        :return: Number of points matching the filter.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return 0
        result = await self._client.count(
            collection_name=collection_name,
            count_filter=query_filter,
        )
        return result.count

    async def store(self, entry: Entry, *, collection_name: str | None = None):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        # Embed the document
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.
        embeddings = await self._embedding_provider.embed_documents([entry.content])

        # Add to Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        payload = {"document": entry.content, METADATA_PATH: entry.metadata}
        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={vector_name: embeddings[0]},
                    payload=payload,
                )
            ],
        )

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.

        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # Embed the query
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.

        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        # Search in Qdrant
        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
        )

        return [
            Entry(
                content=result.payload["document"],
                metadata=result.payload.get("metadata"),
            )
            for result in search_results.points
        ]

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            vector_size = self._embedding_provider.get_vector_size()

            # Use the vector name as defined in the embedding provider
            vector_name = self._embedding_provider.get_vector_name()
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )

            # Create payload indexes if configured

            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )

    async def get_point(
        self,
        collection_name: str,
        ids: list[int | str],
        with_vector: bool = False,
        with_payload: bool = True,
    ) -> list[dict]:
        """
        Retrieve points from a Qdrant collection by their IDs.
        :param collection_name: The name of the collection.
        :param ids: List of point IDs to retrieve.
        :param with_vector: Whether to include vector data.
        :param with_payload: Whether to include payload data.
        :return: List of dicts with id, payload, and optionally vector.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []
        results = await self._client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_payload=with_payload,
            with_vectors=with_vector,
        )
        output = []
        for point in results:
            item = {
                "id": point.id,
                "payload": point.payload or {},
                **({"vector": point.vector} if point.vector else {}),
            }
            # Local Qdrant client returns empty dict {} instead of omitting when with_payload=False
            if not with_payload:
                item.pop("payload", None)
            output.append(item)
        return output

    async def update_points(
        self,
        collection_name: str,
        points_list: list[dict],
    ) -> str:
        """
        Update payloads on specific points by their IDs.
        :param collection_name: The name of the collection.
        :param points_list: List of dicts with 'id' and optional 'payload' keys.
                           E.g., [{'id': 1, 'payload': {'key': 'value'}}, ...]
        :return: A confirmation message.
        """
        ids = [p["id"] for p in points_list]

        # Call set_payload individually per point to avoid issues with local Qdrant client
        # (which doesn't properly handle per-point payload maps when multiple points are given)
        for item in points_list:
            payload = item.get("payload", {})
            await self._client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=[item["id"]],
            )

        return f"Updated payloads on {len(ids)} points in '{collection_name}'"

    async def delete_points(
        self,
        collection_name: str,
        ids: list[int | str],
    ) -> str:
        """
        Delete points from a Qdrant collection by their IDs.
        :param collection_name: The name of the collection.
        :param ids: List of point IDs to delete.
        :return: A confirmation message.
        """
        await self._client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=ids),
        )
        return f"Deleted {len(ids)} points from '{collection_name}'"

    async def delete_vectors(
        self,
        collection_name: str,
        ids: list[int | str],
        vector_names: list[str] | None = None,
    ) -> str:
        """
        Delete specific named vectors from points in a Qdrant collection.
        :param collection_name: The name of the collection.
        :param ids: List of point IDs.
        :param vector_names: Specific vector names to delete; if None, deletes all vectors.
        :return: A confirmation message.
        """
        if vector_names is None:
            # Get actual vector names from collection config
            info = await self._client.get_collection(collection_name)
            vector_names = list(info.config.params.vectors.keys())

        await self._client.delete_vectors(
            collection_name=collection_name,
            vectors=vector_names,
            points=ids,
        )
        label = ", ".join(vector_names) if vector_names else "all"
        return f"Deleted vectors {label} from {len(ids)} points in '{collection_name}'"

    async def update_vectors(
        self,
        collection_name: str,
        points: list[dict],
    ) -> str:
        """
        Update vector values on specific points by their IDs.
        :param collection_name: The name of the collection.
        :param points: List of dicts with 'id' and 'vector' keys.
                      E.g., [{'id': 1, 'vector': [0.1, 0.2, ...]}, ...]
        :return: A confirmation message.
        """
        # Determine the vector name from collection config
        info = await self._client.get_collection(collection_name)
        vectors_config = info.config.params.vectors

        # Handle both named and unnamed vectors
        if isinstance(vectors_config, dict):
            # Named vectors — use the first (and typically only) vector name
            vector_name = next(iter(vectors_config.keys()))
            upsert_points = [
                models.PointStruct(id=p["id"], vector={vector_name: p["vector"]})
                for p in points
            ]
        elif vectors_config is not None and not isinstance(vectors_config, dict):
            # Unnamed single vector config
            upsert_points = [
                models.PointStruct(id=p["id"], vector=p["vector"])
                for p in points
            ]
        else:
            # No vectors configured — should not happen in practice
            raise ValueError(f"No vector configuration found in collection '{collection_name}'")

        await self._client.upsert(
            collection_name=collection_name,
            points=upsert_points,
        )

        return f"Updated vectors on {len(points)} points in '{collection_name}'"

    async def batch_update(
        self,
        collection_name: str,
        operations: list[dict],
    ) -> str:
        """
        Execute multiple point operations in a single request.

        Each operation is a dict with 'operation_type' and type-specific fields:
          - {operation_type: "upsert", points: [PointStruct-like dicts]}
          - {operation_type: "delete_points", ids: [...]}
          - {operation_type: "set_payload", payload: {...}, ids: [...]}
          - {operation_type: "update_vectors", points: [{id, vector}]}
          - {operation_type: "delete_vectors", ids: [...], vector_names: [...]}

        :param collection_name: The name of the collection.
        :param operations: List of operation dicts.
        :return: A confirmation message.
        """
        op_list = []
        for op in operations:
            op_type = op["operation_type"]

            if op_type == "upsert":
                point_structs = []
                for p in op["points"]:
                    ps = models.PointStruct(id=p["id"], vector=p["vector"])
                    if "payload" in p:
                        ps.payload = p["payload"]
                    point_structs.append(ps)
                op_list.append(
                    models.UpsertOperation(upsert=models.PointsList(points=point_structs))
                )

            elif op_type == "delete_points":
                ids = op.get("ids") or op.get("points_selector", {}).get("points")
                if ids:
                    op_list.append(
                        models.DeleteOperation(delete=models.PointsList(points=ids))
                    )

            elif op_type == "set_payload":
                payload = op["payload"]
                ids = op.get("ids")
                if ids:
                    # set_payload expects a mapping of id -> payload, or a single payload for multiple ids
                    if len(ids) == 1 and isinstance(payload, dict) and "id" not in payload:
                        # Single id with payload dict -> use it directly
                        op_list.append(
                            models.SetPayloadOperation(set_payload=models.SetPayload(payload=payload, points=[ids[0]]))
                        )
                    else:
                        op_list.append(
                            models.SetPayloadOperation(set_payload=models.SetPayload(payload=payload, points=ids))
                        )

            elif op_type == "update_vectors":
                vectors = [models.UpdateVectors(id=p["id"], vector=p["vector"]) for p in op["points"]]
                op_list.append(
                    models.UpdateVectorsOperation(update_vectors=models.PointsList(points=vectors))
                )

            elif op_type == "delete_vectors":
                ids = op.get("ids") or op.get("points_selector", {}).get("points")
                vector_names = op.get("vector_names", [])
                if ids:
                    op_list.append(
                        models.DeleteVectorsOperation(delete_vectors=models.DeleteVectors(points=ids, vector_names=vector_names))
                    )

        await self._client.batch_update_points(
            collection_name=collection_name,
            update_operations=op_list,
        )

        return f"Executed {len(operations)} batch operations in '{collection_name}'"

    async def recommend(
        self,
        collection_name: str,
        positive: list[int | str] | None = None,
        negative: list[int | str] | None = None,
        query_filter: models.Filter | None = None,
        limit: int = 10,
        with_vector: bool = False,
        with_payload: bool = True,
        score_threshold: float | None = None,
        using: str | None = None,
    ) -> list[dict]:
        """
        Recommend similar points based on positive/negative example IDs.

        Uses the 'look-alike' search pattern — finds points similar to positive examples
        and dissimilar to negative examples.
        :param collection_name: The name of the collection.
        :param positive: Point IDs to find similar to.
        :param negative: Point IDs to avoid.
        :param query_filter: Optional filter to apply.
        :param limit: Maximum number of points to return.
        :param with_vector: Whether to include vector data.
        :param with_payload: Whether to include payload data.
        :param score_threshold: Minimum similarity score (0.0–1.0).
        :param using: Name of the vector params to use for search.
        :return: List of matching points with scores.
        """
        # Check if collection exists
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # If using is not specified, determine it from collection config
        if using is None:
            info = await self._client.get_collection(collection_name)
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict) and vectors_config:
                using = next(iter(vectors_config.keys()))

        recommend_query = models.RecommendQuery(
            recommend=models.RecommendInput(
                positive=positive,
                negative=negative,
            )
        )

        results = await self._client.query_points(
            collection_name=collection_name,
            query=recommend_query,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            using=using,
            with_payload=with_payload,
            with_vectors=with_vector,
        )

        return [
            {
                "id": point.id,
                "payload": point.payload or {},
                "score": point.score,
                **({"vector": point.vector} if point.vector else {}),
            }
            for point in results.points
        ]
