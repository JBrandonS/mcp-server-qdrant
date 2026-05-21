import json
import logging
from typing import Annotated, Any, Optional

from fastmcp import Context, FastMCP
from pydantic import Field
from qdrant_client import models

from mcp_server_qdrant.common.filters import make_indexes
from mcp_server_qdrant.common.func_tools import make_partial_function
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings

        if embedding_provider_settings and embedding_provider:
            raise ValueError(
                "Cannot provide both embedding_provider_settings and embedding_provider"
            )

        if not embedding_provider_settings and not embedding_provider:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )

        self.embedding_provider_settings: Optional[EmbeddingProviderSettings] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        else:
            self.embedding_provider_settings = None
            self.embedding_provider = embedding_provider

        assert self.embedding_provider is not None, "Embedding provider is required"

        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
            make_indexes(qdrant_settings.filterable_fields_dict()),
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[
                str, Field(description="The collection to store the information in")
            ],
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="Extra metadata stored along with memorised information. Any json is accepted."
                ),
            ] = None,
        ) -> str:
            """
            Store some information in Qdrant.
            :param ctx: The context for the request.
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored.
            """
            await ctx.debug(f"Storing information {information} in Qdrant")

            entry = Entry(content=information, metadata=metadata)

            await self.qdrant_connector.store(entry, collection_name=collection_name)
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str, Field(description="The collection to search in")
            ],
            query_filter: ArbitraryFilter | None = None,
        ) -> list[str] | None:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :param query_filter: The filter to apply to the query.
            :return: A list of entries found or None.
            """

            await ctx.debug(f"Query filter: {query_filter}")

            query_filter = models.Filter(**query_filter) if query_filter else None

            await ctx.debug(f"Finding results for query {query}")

            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
                query_filter=query_filter,
            )
            if not entries:
                return None
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        find_foo = find
        store_foo = store

        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})

        if self.qdrant_settings.collection_name:
            find_foo = make_partial_function(
                find_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            store_foo = make_partial_function(
                store_foo, {"collection_name": self.qdrant_settings.collection_name}
            )

        self.tool(
            find_foo,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )

        if not self.qdrant_settings.read_only:
            self.tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )

        async def list_collections(
            ctx: Context,
        ) -> list[str]:
            """
            List all collections in the Qdrant server.
            :param ctx: The context for the request.
            :return: A list of collection names.
            """
            await ctx.debug("Listing all collections")
            return await self.qdrant_connector.list_collections()

        self.tool(
            list_collections,
            name="qdrant-list-collections",
            description="List all collections in the Qdrant server",
        )

        async def get_collection_info(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The name of the collection")
            ],
        ) -> dict:
            """
            Get detailed information about a Qdrant collection including size, segments, and configuration.
            :param ctx: The context for the request.
            :param collection_name: The name of the collection.
            :return: A dict with stats and configuration.
            """
            await ctx.debug(f"Getting info for collection {collection_name}")
            return await self.qdrant_connector.get_collection_info(collection_name)

        self.tool(
            get_collection_info,
            name="qdrant-get-collection-info",
            description="Get detailed information about a Qdrant collection including size, segments, and configuration",
        )

        if not self.qdrant_settings.read_only:
            async def create_collection(
                ctx: Context,
                collection_name: Annotated[
                    str, Field(description="Name for the new collection")
                ],
                vector_size: Annotated[
                    int, Field(description="Vector dimension matching your embedding model (e.g., 384 for all-MiniLM-L6-v2)")
                ],
                distance: Annotated[
                    str, Field(description="Distance metric: Cosine, Euclid, Dot, or Manhattan (default: Cosine)")
                ] = "Cosine",
            ) -> str:
                """
                Create a new Qdrant collection with vector configuration.
                :param ctx: The context for the request.
                :param collection_name: Name for the new collection.
                :param vector_size: Vector dimension matching your embedding model.
                :param distance: Distance metric to use.
                :return: A confirmation message.
                """
                await ctx.debug(f"Creating collection {collection_name} with vector_size={vector_size}, distance={distance}")
                return await self.qdrant_connector.create_collection(
                    collection_name, vector_size, distance
                )

            self.tool(
                create_collection,
                name="qdrant-create-collection",
                description="Create a new Qdrant collection. Requires vector size (matching your embedding model) and optional distance metric (Cosine, Euclid, Dot, Manhattan)",
            )

            async def update_collection(
                ctx: Context,
                collection_name: Annotated[
                    str, Field(description="The name of the collection")
                ],
                optimizer_config: Annotated[
                    dict | None, Field(description="JSON object with optimizer settings like indexing_threshold, max_segment_size, deleted_threshold, vacuum_min_vector_number", default=None)
                ] = None,
                replication_factor: Annotated[
                    int | None, Field(description="Number of copies of each shard", default=None)
                ] = None,
                write_consistency_factor: Annotated[
                    int | None, Field(description="How many replicas must acknowledge writes", default=None)
                ] = None,
            ) -> str:
                """
                Update Qdrant collection settings like optimizer thresholds and replication factors.
                :param ctx: The context for the request.
                :param collection_name: The name of the collection.
                :param optimizer_config: Optional optimizer settings.
                :param replication_factor: Optional number of shard copies.
                :param write_consistency_factor: Optional write consistency.
                :return: A confirmation message.
                """
                await ctx.debug(f"Updating collection {collection_name}")
                return await self.qdrant_connector.update_collection(
                    collection_name,
                    optimizer_config=optimizer_config,
                    replication_factor=replication_factor,
                    write_consistency_factor=write_consistency_factor,
                )

            self.tool(
                update_collection,
                name="qdrant-update-collection",
                description="Update Qdrant collection settings like optimizer thresholds and replication factors",
            )

            async def delete_collection(
                ctx: Context,
                collection_name: Annotated[
                    str, Field(description="The name of the collection to delete")
                ],
            ) -> str:
                """
                Delete a Qdrant collection and all its associated data. This operation is irreversible.
                :param ctx: The context for the request.
                :param collection_name: The name of the collection to delete.
                :return: A confirmation message.
                """
                await ctx.debug(f"Deleting collection {collection_name}")
                return await self.qdrant_connector.delete_collection(collection_name)

            self.tool(
                delete_collection,
                name="qdrant-delete-collection",
                description="Delete a Qdrant collection and all its associated data. This operation is irreversible.",
            )

        async def list_points(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The name of the collection")
            ],
            query_filter: ArbitraryFilter | None = None,
            limit: Annotated[
                int, Field(description="Maximum number of points to return (default: 10)")
            ] = 10,
            offset: Annotated[
                str | int | None, Field(description="Offset for pagination")
            ] = None,
            with_vector: Annotated[
                bool | list[str] | None, Field(description="Include vector data in results (default: False)")
            ] = None,
            with_payload: Annotated[
                bool | list[str] | None, Field(description="Include payload data in results (default: True)")
            ] = None,
        ) -> dict:
            """
            List points from a Qdrant collection with optional filtering and pagination.
            :param ctx: The context for the request.
            :param collection_name: The name of the collection.
            :param query_filter: Optional JSON filter object.
            :param limit: Maximum number of points to return.
            :param offset: Offset for pagination.
            :param with_vector: Whether to include vector data.
            :param with_payload: Whether to include payload data.
            :return: Dict with 'points' list and 'next_offset'.
            """
            await ctx.debug(f"Listing points in collection {collection_name}")
            filter_obj = models.Filter(**query_filter) if query_filter else None
            return await self.qdrant_connector.list_points(
                collection_name,
                query_filter=filter_obj,
                limit=limit,
                offset=offset,
                with_vector=with_vector if with_vector is not None else False,
                with_payload=with_payload if with_payload is not None else True,
            )

        self.tool(
            list_points,
            name="qdrant-list-points",
            description="List points from a Qdrant collection with optional filtering and pagination. Returns paginated results with next_offset for continuation.",
        )

        async def count_points(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The name of the collection")
            ],
            query_filter: ArbitraryFilter | None = None,
        ) -> int:
            """
            Count points in a Qdrant collection, optionally filtered.
            :param ctx: The context for the request.
            :param collection_name: The name of the collection.
            :param query_filter: Optional JSON filter object.
            :return: Number of points matching the filter.
            """
            await ctx.debug(f"Counting points in collection {collection_name}")
            filter_obj = models.Filter(**query_filter) if query_filter else None
            return await self.qdrant_connector.count_points(
                collection_name, query_filter=filter_obj
            )

        self.tool(
            count_points,
            name="qdrant-count-points",
            description="Count the number of points in a Qdrant collection, optionally filtered.",
        )

        async def get_point(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The name of the collection")
            ],
            ids: Annotated[
                list, Field(description="List of point IDs to retrieve")
            ],
            with_vector: Annotated[
                bool, Field(description="Include vector data in results (default: False)")
            ] = False,
            with_payload: Annotated[
                bool, Field(description="Include payload data in results (default: True)")
            ] = True,
        ) -> list:
            """
            Retrieve points from a Qdrant collection by their IDs.
            :param ctx: The context for the request.
            :param collection_name: The name of the collection.
            :param ids: List of point IDs to retrieve.
            :param with_vector: Whether to include vector data.
            :param with_payload: Whether to include payload data.
            :return: List of dicts with id, payload, and optionally vector.
            """
            await ctx.debug(f"Getting points {ids} from collection {collection_name}")
            return await self.qdrant_connector.get_point(
                collection_name, ids, with_vector=with_vector, with_payload=with_payload
            )

        self.tool(
            get_point,
            name="qdrant-get-point",
            description="Retrieve points from a Qdrant collection by their IDs.",
        )

        if not self.qdrant_settings.read_only:
            async def update_points(
                ctx: Context,
                collection_name: Annotated[
                    str, Field(description="The name of the collection")
                ],
                points_list: Annotated[
                    list, Field(description="List of dicts with 'id' and optional 'payload' keys. E.g., [{'id': 1, 'payload': {'key': 'value'}}, ...]")
                ],
            ) -> str:
                """
                Update payloads on specific points by their IDs.
                :param ctx: The context for the request.
                :param collection_name: The name of the collection.
                :param points_list: List of dicts with 'id' and optional 'payload' keys.
                :return: A confirmation message.
                """
                await ctx.debug(f"Updating payloads on points in collection {collection_name}")
                return await self.qdrant_connector.update_points(
                    collection_name, points_list
                )

            self.tool(
                update_points,
                name="qdrant-update-points",
                description="Update payloads on specific points by their IDs.",
            )

            async def delete_points(
                ctx: Context,
                collection_name: Annotated[
                    str, Field(description="The name of the collection")
                ],
                ids: Annotated[
                    list, Field(description="List of point IDs to delete")
                ],
            ) -> str:
                """
                Delete points from a Qdrant collection by their IDs.
                :param ctx: The context for the request.
                :param collection_name: The name of the collection.
                :param ids: List of point IDs to delete.
                :return: A confirmation message.
                """
                await ctx.debug(f"Deleting points {ids} from collection {collection_name}")
                return await self.qdrant_connector.delete_points(
                    collection_name, ids
                )

            self.tool(
                delete_points,
                name="qdrant-delete-points",
                description="Delete points from a Qdrant collection by their IDs.",
            )

            async def delete_vectors(
                ctx: Context,
                collection_name: Annotated[
                    str, Field(description="The name of the collection")
                ],
                ids: Annotated[
                    list, Field(description="List of point IDs")
                ],
                vector_names: Annotated[
                    list[str] | None, Field(description="Specific vector names to delete; if None, deletes all vectors", default=None)
                ] = None,
            ) -> str:
                """
                Delete specific named vectors from points in a Qdrant collection.
                :param ctx: The context for the request.
                :param collection_name: The name of the collection.
                :param ids: List of point IDs.
                :param vector_names: Specific vector names to delete; if None, deletes all vectors.
                :return: A confirmation message.
                """
                await ctx.debug(f"Deleting vectors from points in collection {collection_name}")
                return await self.qdrant_connector.delete_vectors(
                    collection_name, ids, vector_names=vector_names
                )

            self.tool(
                delete_vectors,
                name="qdrant-delete-vectors",
                description="Delete specific named vectors from points in a Qdrant collection.",
            )

            async def update_vectors(
                ctx: Context,
                collection_name: Annotated[
                    str, Field(description="The name of the collection")
                ],
                points: Annotated[
                    list, Field(description="List of dicts with 'id' and 'vector' keys. E.g., [{'id': 1, 'vector': [0.1, 0.2, ...]}, ...]")
                ],
            ) -> str:
                """
                Update vector values on specific points by their IDs.
                :param ctx: The context for the request.
                :param collection_name: The name of the collection.
                :param points: List of dicts with 'id' and 'vector' keys.
                :return: A confirmation message.
                """
                await ctx.debug(f"Updating vectors in collection {collection_name}")
                return await self.qdrant_connector.update_vectors(
                    collection_name, points
                )

            self.tool(
                update_vectors,
                name="qdrant-update-vectors",
                description="Update vector values on specific points by their IDs.",
            )

            async def batch_update(
                ctx: Context,
                collection_name: Annotated[
                    str, Field(description="The name of the collection")
                ],
                operations: Annotated[
                    list, Field(description="List of operation dicts. Each operation has 'operation_type' and type-specific fields. Supported types: upsert, delete_points, set_payload, update_vectors, delete_vectors")
                ],
            ) -> str:
                """
                Execute multiple point operations in a single request.
                :param ctx: The context for the request.
                :param collection_name: The name of the collection.
                :param operations: List of operation dicts.
                :return: A confirmation message.
                """
                await ctx.debug(f"Batch updating collection {collection_name}")
                return await self.qdrant_connector.batch_update(
                    collection_name, operations
                )

            self.tool(
                batch_update,
                name="qdrant-batch-update",
                description="Execute multiple point operations (upsert, delete_points, set_payload, update_vectors, delete_vectors) in a single request.",
            )

        async def recommend(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The name of the collection")
            ],
            positive: Annotated[
                list | None, Field(description="Point IDs to find similar to", default=None)
            ] = None,
            negative: Annotated[
                list | None, Field(description="Point IDs to avoid", default=None)
            ] = None,
            query_filter: ArbitraryFilter | None = None,
            limit: Annotated[
                int, Field(description="Maximum number of points to return (default: 10)")
            ] = 10,
            with_vector: Annotated[
                bool, Field(description="Include vector data in results (default: False)")
            ] = False,
            with_payload: Annotated[
                bool, Field(description="Include payload data in results (default: True)")
            ] = True,
            score_threshold: Annotated[
                float | None, Field(description="Minimum similarity score (0.0-1.0)", default=None)
            ] = None,
            using: Annotated[
                str | None, Field(description="Name of the vector params to use for search", default=None)
            ] = None,
        ) -> list:
            """
            Recommend similar points based on positive/negative example IDs.
            Uses the 'look-alike' search pattern.
            :param ctx: The context for the request.
            :param collection_name: The name of the collection.
            :param positive: Point IDs to find similar to.
            :param negative: Point IDs to avoid.
            :param query_filter: Optional filter to apply.
            :param limit: Maximum number of points to return.
            :param with_vector: Whether to include vector data.
            :param with_payload: Whether to include payload data.
            :param score_threshold: Minimum similarity score (0.0-1.0).
            :param using: Name of the vector params to use for search.
            :return: List of matching points with scores.
            """
            await ctx.debug(f"Recommending points in collection {collection_name}")
            filter_obj = models.Filter(**query_filter) if query_filter else None
            return await self.qdrant_connector.recommend(
                collection_name,
                positive=positive,
                negative=negative,
                query_filter=filter_obj,
                limit=limit,
                with_vector=with_vector,
                with_payload=with_payload,
                score_threshold=score_threshold,
                using=using,
            )

        self.tool(
            recommend,
            name="qdrant-recommend",
            description="Recommend similar points based on positive/negative example IDs using the look-alike search pattern.",
        )
