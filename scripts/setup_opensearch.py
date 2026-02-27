"""Complete OpenSearch Serverless (AOSS) infrastructure setup.

Sets up everything needed for the RAG Bot v2 OpenSearch infrastructure:

  Phase 1 - IAM: connector role + Bedrock permissions
  Phase 2 - AOSS: encryption, network, data access policies + collection
  Phase 3 - OpenSearch: ML connector, model, pipelines, index

Idempotent - safely skips resources that already exist.

Usage::

    conda run --prefix .venv python scripts/setup_opensearch.py

Prerequisites:
    - AWS credentials configured (env vars, profile, or instance role)
    - The calling identity must have permissions to create IAM roles and
      AOSS resources (typically an admin or power-user role)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AossSetupConfig:
    """Configuration for the AOSS infrastructure setup.

    All values have sensible defaults matching the RAG Bot v2 project.
    Override via environment variables where noted.
    """

    # AWS identity (auto-detected)
    region: str
    account_id: str
    caller_arn: str

    # IAM
    connector_role_name: str
    embedding_model_arn: str

    # AOSS collection
    collection_name: str
    encryption_policy_name: str
    network_policy_name: str
    data_access_policy_name: str

    # OpenSearch index / pipelines
    index_name: str
    ingest_pipeline_name: str
    search_pipeline_name: str
    embedding_field: str
    embedding_dim: int

    # ML connector / model display names
    connector_display_name: str
    model_display_name: str

    # Additional principals for the data access policy
    additional_principals: list[str] = field(default_factory=list)

    # Polling
    collection_poll_interval_s: int = 10
    collection_poll_max_attempts: int = 60
    model_poll_interval_s: int = 5
    model_poll_max_attempts: int = 60

    # Delay between Phase 2 and Phase 3 (eventual consistency)
    phase_transition_delay_s: int = 30

    @staticmethod
    def from_env() -> AossSetupConfig:
        """Build config from environment variables with project defaults."""

        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        account_id = identity["Account"]
        caller_arn = identity["Arn"]

        region = (
            os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "us-west-2"
        )

        collection_name = (
            os.getenv("OPENSEARCH_COLLECTION_NAME") or "ragbot-v2-collection"
        )
        index_name = os.getenv("OPENSEARCH_INDEX_NAME") or "sagemaker-docs"

        dim_raw = os.getenv("BEDROCK_EMBEDDING_DIM") or "1024"
        embedding_dim = int(dim_raw)

        embedding_model_arn = (
            f"arn:aws:bedrock:{region}::foundation-model/"
            "amazon.titan-embed-text-v2:0"
        )

        additional: list[str] = []
        raw_principals = os.getenv("AOSS_ADDITIONAL_PRINCIPALS", "")
        if raw_principals.strip():
            additional = [
                p.strip() for p in raw_principals.split(",") if p.strip()
            ]

        return AossSetupConfig(
            region=region,
            account_id=account_id,
            caller_arn=caller_arn,
            connector_role_name="opensearch-bedrock-connector-role",
            embedding_model_arn=embedding_model_arn,
            collection_name=collection_name,
            encryption_policy_name=f"{collection_name}-encryption",
            network_policy_name=f"{collection_name}-network",
            data_access_policy_name=f"{collection_name}-data-access",
            index_name=index_name,
            # Must match constants in app/services/opensearch_service.py
            ingest_pipeline_name="sagemaker-docs-ingest-pipeline",
            search_pipeline_name="sagemaker-docs-search-pipeline",
            embedding_field="content_embedding",
            embedding_dim=embedding_dim,
            connector_display_name="Amazon Bedrock - Titan Embed v2",
            model_display_name="Bedrock Titan Embed v2",
            additional_principals=additional,
        )


# ---------------------------------------------------------------------------
# Setup runner
# ---------------------------------------------------------------------------


class AossSetupRunner:
    """Orchestrates the complete AOSS infrastructure setup.

    Each public method corresponds to one phase and is idempotent.
    """

    def __init__(self, config: AossSetupConfig) -> None:
        self._config = config
        self._iam = boto3.client("iam", region_name=config.region)
        self._aoss = boto3.client(
            "opensearchserverless", region_name=config.region,
        )

        # Populated during execution
        self._collection_endpoint: Optional[str] = None
        self._connector_id: Optional[str] = None
        self._model_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Phase 1: IAM
    # ------------------------------------------------------------------

    def setup_iam(self) -> None:
        """Create IAM roles and policies for AOSS + Bedrock integration."""
        logger.info("Phase 1: IAM Setup")
        self._create_connector_role()
        self._attach_connector_permissions()
        logger.info("Phase 1 complete.")

    def _create_connector_role(self) -> None:
        """Create the IAM role that AOSS ML assumes to call Bedrock."""
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "aoss.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                },
            ],
        }
        try:
            self._iam.create_role(
                RoleName=self._config.connector_role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description=(
                    "Role for OpenSearch ML to invoke Bedrock embeddings"
                ),
            )
            logger.info(
                "Created role: %s", self._config.connector_role_name,
            )
        except self._iam.exceptions.EntityAlreadyExistsException:
            logger.info(
                "Role already exists: %s (skipping)",
                self._config.connector_role_name,
            )

    def _attach_connector_permissions(self) -> None:
        """Attach the Bedrock invoke-model policy to the connector role."""
        policy_doc = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "bedrock:InvokeModel",
                    "Resource": self._config.embedding_model_arn,
                },
            ],
        }
        self._iam.put_role_policy(
            RoleName=self._config.connector_role_name,
            PolicyName="bedrock-invoke-policy",
            PolicyDocument=json.dumps(policy_doc),
        )
        logger.info("Attached bedrock-invoke-policy to connector role.")

    # ------------------------------------------------------------------
    # Phase 2: AOSS infrastructure
    # ------------------------------------------------------------------

    def setup_aoss_infrastructure(self) -> str:
        """Create AOSS policies, collection, and data access rules.

        Returns:
            The collection endpoint URL.
        """
        logger.info("Phase 2: AOSS Infrastructure")
        self._create_encryption_policy()
        self._create_network_policy()
        self._create_collection()
        self._wait_for_collection_active()
        self._create_data_access_policy()
        logger.info(
            "Phase 2 complete. Endpoint: %s", self._collection_endpoint,
        )
        return self._collection_endpoint  # type: ignore[return-value]

    def _create_encryption_policy(self) -> None:
        policy_body = json.dumps(
            {
                "Rules": [
                    {
                        "ResourceType": "collection",
                        "Resource": [
                            f"collection/{self._config.collection_name}",
                        ],
                    },
                ],
                "AWSOwnedKey": True,
            },
        )
        try:
            self._aoss.create_security_policy(
                name=self._config.encryption_policy_name,
                type="encryption",
                policy=policy_body,
                description=(
                    f"Encryption policy for {self._config.collection_name}"
                ),
            )
            logger.info(
                "Created encryption policy: %s",
                self._config.encryption_policy_name,
            )
        except self._aoss.exceptions.ConflictException:
            logger.info(
                "Encryption policy already exists: %s (skipping)",
                self._config.encryption_policy_name,
            )

    def _create_network_policy(self) -> None:
        policy_body = json.dumps(
            [
                {
                    "Rules": [
                        {
                            "ResourceType": "collection",
                            "Resource": [
                                f"collection/{self._config.collection_name}",
                            ],
                        },
                        {
                            "ResourceType": "dashboard",
                            "Resource": [
                                f"collection/{self._config.collection_name}",
                            ],
                        },
                    ],
                    "AllowFromPublic": True,
                },
            ],
        )
        try:
            self._aoss.create_security_policy(
                name=self._config.network_policy_name,
                type="network",
                policy=policy_body,
                description=(
                    f"Network policy for {self._config.collection_name}"
                ),
            )
            logger.info(
                "Created network policy: %s",
                self._config.network_policy_name,
            )
        except self._aoss.exceptions.ConflictException:
            logger.info(
                "Network policy already exists: %s (skipping)",
                self._config.network_policy_name,
            )

    def _create_collection(self) -> None:
        try:
            resp = self._aoss.create_collection(
                name=self._config.collection_name,
                type="VECTORSEARCH",
                description="RAG Bot v2 vector search collection",
            )
            coll_id = resp["createCollectionDetail"]["id"]
            logger.info(
                "Created collection: %s (id: %s)",
                self._config.collection_name,
                coll_id,
            )
        except self._aoss.exceptions.ConflictException:
            logger.info(
                "Collection already exists: %s (skipping)",
                self._config.collection_name,
            )

    def _wait_for_collection_active(self) -> None:
        """Poll until the collection reaches ACTIVE state."""
        cfg = self._config
        for attempt in range(1, cfg.collection_poll_max_attempts + 1):
            resp = self._aoss.batch_get_collection(
                names=[cfg.collection_name],
            )
            details = resp.get("collectionDetails", [])
            if not details:
                raise RuntimeError(
                    f"Collection not found: {cfg.collection_name}",
                )

            status = details[0].get("status")
            logger.info(
                "Collection status (attempt %d/%d): %s",
                attempt,
                cfg.collection_poll_max_attempts,
                status,
            )

            if status == "ACTIVE":
                self._collection_endpoint = details[0]["collectionEndpoint"]
                return

            if status in ("FAILED", "DELETING"):
                raise RuntimeError(
                    f"Collection in terminal state: {status}",
                )

            time.sleep(cfg.collection_poll_interval_s)

        timeout = cfg.collection_poll_max_attempts * cfg.collection_poll_interval_s
        raise RuntimeError(
            f"Collection did not reach ACTIVE state after {timeout}s",
        )

    def _create_data_access_policy(self) -> None:
        """Create or update the data access policy with all principals."""
        connector_role_arn = (
            f"arn:aws:iam::{self._config.account_id}"
            f":role/{self._config.connector_role_name}"
        )

        principals: list[str] = [self._config.caller_arn, connector_role_arn]
        for p in self._config.additional_principals:
            if p and p not in principals:
                principals.append(p)

        policy_body = json.dumps(
            [
                {
                    "Rules": [
                        {
                            "ResourceType": "collection",
                            "Resource": [
                                f"collection/{self._config.collection_name}",
                            ],
                            "Permission": [
                                "aoss:CreateCollectionItems",
                                "aoss:DeleteCollectionItems",
                                "aoss:UpdateCollectionItems",
                                "aoss:DescribeCollectionItems",
                            ],
                        },
                        {
                            "ResourceType": "index",
                            "Resource": [
                                f"index/{self._config.collection_name}/*",
                            ],
                            "Permission": [
                                "aoss:CreateIndex",
                                "aoss:DeleteIndex",
                                "aoss:UpdateIndex",
                                "aoss:DescribeIndex",
                                "aoss:ReadDocument",
                                "aoss:WriteDocument",
                            ],
                        },
                    ],
                    "Principal": principals,
                    "Description": (
                        f"Data access for {self._config.collection_name}"
                    ),
                },
            ],
        )

        try:
            self._aoss.create_access_policy(
                name=self._config.data_access_policy_name,
                type="data",
                policy=policy_body,
                description=(
                    f"Data access policy for {self._config.collection_name}"
                ),
            )
            logger.info(
                "Created data access policy: %s",
                self._config.data_access_policy_name,
            )
        except self._aoss.exceptions.ConflictException:
            logger.info(
                "Data access policy exists, updating principals: %s",
                self._config.data_access_policy_name,
            )
            existing = self._aoss.get_access_policy(
                name=self._config.data_access_policy_name,
                type="data",
            )
            version = existing["accessPolicyDetail"]["policyVersion"]
            self._aoss.update_access_policy(
                name=self._config.data_access_policy_name,
                type="data",
                policy=policy_body,
                policyVersion=version,
                description=(
                    f"Data access policy for {self._config.collection_name}"
                ),
            )
            logger.info("Updated data access policy with current principals.")

    # ------------------------------------------------------------------
    # Phase 3: OpenSearch resources (via data-plane API)
    # ------------------------------------------------------------------

    def setup_opensearch_resources(self) -> None:
        """Create ML connector, model, pipelines, and index."""
        logger.info("Phase 3: OpenSearch Resource Setup")

        if not self._collection_endpoint:
            self._resolve_collection_endpoint()

        logger.info(
            "Waiting %ds for data access policy propagation...",
            self._config.phase_transition_delay_s,
        )
        time.sleep(self._config.phase_transition_delay_s)

        client = self._build_opensearch_client()

        self._connector_id = self._create_or_get_ml_connector(client)
        self._model_id = self._create_or_get_ml_model(client)
        self._deploy_model(client)
        self._create_ingest_pipeline(client)
        self._create_search_pipeline(client)
        self._create_index(client)

        logger.info("Phase 3 complete.")

    def _resolve_collection_endpoint(self) -> None:
        """Fetch the endpoint from an existing ACTIVE collection."""
        resp = self._aoss.batch_get_collection(
            names=[self._config.collection_name],
        )
        details = resp.get("collectionDetails", [])
        if not details:
            raise RuntimeError(
                f"Collection not found: {self._config.collection_name}",
            )
        if details[0]["status"] != "ACTIVE":
            raise RuntimeError(
                f"Collection not active: {details[0]['status']}",
            )
        self._collection_endpoint = details[0]["collectionEndpoint"]

    def _build_opensearch_client(self) -> OpenSearch:
        """Build an opensearch-py client with SigV4 auth.

        Mirrors the pattern in app/services/opensearch_service.py.
        """
        credentials = boto3.Session().get_credentials()
        if credentials is None:
            raise RuntimeError("AWS credentials not available")

        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            self._config.region,
            "aoss",
            session_token=credentials.token,
        )

        host = (
            self._collection_endpoint
            .replace("https://", "")
            .replace("http://", "")
            .rstrip("/")
        )

        return OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30,
        )

    # -- ML connector --------------------------------------------------

    def _create_or_get_ml_connector(self, client: OpenSearch) -> str:
        """Create the Bedrock ML connector, or return existing ID."""
        # Search for existing connector by name
        try:
            resp = client.transport.perform_request(
                "POST",
                "/_plugins/_ml/connectors/_search",
                body={
                    "query": {
                        "match": {
                            "name": self._config.connector_display_name,
                        },
                    },
                },
            )
            for hit in resp.get("hits", {}).get("hits", []):
                if hit.get("_source", {}).get("name") == (
                    self._config.connector_display_name
                ):
                    connector_id = hit["_id"]
                    logger.info(
                        "ML connector already exists: %s (id: %s)",
                        self._config.connector_display_name,
                        connector_id,
                    )
                    return connector_id
        except Exception:
            logger.debug("Connector search returned no results.")

        # Create connector
        connector_role_arn = (
            f"arn:aws:iam::{self._config.account_id}"
            f":role/{self._config.connector_role_name}"
        )
        region = self._config.region

        connector_body: dict[str, Any] = {
            "name": self._config.connector_display_name,
            "description": (
                "Connector for Amazon Titan Text Embeddings V2 via Bedrock"
            ),
            "version": 1,
            "protocol": "aws_sigv4",
            "parameters": {
                "region": region,
                "service_name": "bedrock",
            },
            "credential": {
                "roleArn": connector_role_arn,
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": (
                        f"https://bedrock-runtime.{region}.amazonaws.com"
                        "/model/amazon.titan-embed-text-v2:0/invoke"
                    ),
                    "headers": {
                        "content-type": "application/json",
                        "x-amz-content-sha256": "required",
                    },
                    "request_body": (
                        '{ "inputText": "${parameters.inputText}" }'
                    ),
                    "pre_process_function": _CONNECTOR_PRE_PROCESS,
                    "post_process_function": _CONNECTOR_POST_PROCESS,
                },
            ],
        }

        resp = client.transport.perform_request(
            "POST",
            "/_plugins/_ml/connectors/_create",
            body=connector_body,
        )
        connector_id = resp["connector_id"]
        logger.info(
            "Created ML connector: %s (id: %s)",
            self._config.connector_display_name,
            connector_id,
        )
        return connector_id

    # -- ML model ------------------------------------------------------

    def _create_or_get_ml_model(self, client: OpenSearch) -> str:
        """Register the ML model, or return existing ID."""
        # Search for existing model by name
        try:
            resp = client.transport.perform_request(
                "POST",
                "/_plugins/_ml/models/_search",
                body={
                    "query": {
                        "match": {
                            "name": self._config.model_display_name,
                        },
                    },
                },
            )
            for hit in resp.get("hits", {}).get("hits", []):
                if hit.get("_source", {}).get("name") == (
                    self._config.model_display_name
                ):
                    model_id = hit["_id"]
                    logger.info(
                        "ML model already exists: %s (id: %s)",
                        self._config.model_display_name,
                        model_id,
                    )
                    return model_id
        except Exception:
            logger.debug("Model search returned no results.")

        register_body = {
            "name": self._config.model_display_name,
            "function_name": "remote",
            "description": (
                "Titan Text Embedding V2 via Bedrock connector"
            ),
            "connector_id": self._connector_id,
        }

        resp = client.transport.perform_request(
            "POST",
            "/_plugins/_ml/models/_register",
            body=register_body,
        )

        model_id = resp.get("model_id")
        task_id = resp.get("task_id")

        if model_id:
            logger.info(
                "Registered ML model: %s (id: %s)",
                self._config.model_display_name,
                model_id,
            )
            return model_id

        if task_id:
            logger.info(
                "Model registration task: %s — polling for completion...",
                task_id,
            )
            return self._wait_for_model_registration(client, task_id)

        raise RuntimeError(f"Unexpected register response: {resp}")

    def _wait_for_model_registration(
        self, client: OpenSearch, task_id: str,
    ) -> str:
        cfg = self._config
        for attempt in range(1, cfg.model_poll_max_attempts + 1):
            resp = client.transport.perform_request(
                "GET", f"/_plugins/_ml/tasks/{task_id}",
            )
            state = resp.get("state")
            logger.info(
                "Registration task state (attempt %d): %s", attempt, state,
            )
            if state == "COMPLETED":
                return resp["model_id"]
            if state in ("FAILED", "CANCELLED"):
                raise RuntimeError(
                    f"Model registration failed: {resp}",
                )
            time.sleep(cfg.model_poll_interval_s)

        raise RuntimeError("Model registration timed out")

    # -- Deploy model --------------------------------------------------

    def _deploy_model(self, client: OpenSearch) -> None:
        """Deploy the model. Skips if already deployed."""
        try:
            resp = client.transport.perform_request(
                "GET", f"/_plugins/_ml/models/{self._model_id}",
            )
            state = resp.get("model_state")
            if state == "DEPLOYED":
                logger.info("Model already deployed: %s", self._model_id)
                return
            if state == "DEPLOYING":
                logger.info("Model is deploying, waiting...")
                self._wait_for_model_deployed(client)
                return
        except Exception:
            pass

        client.transport.perform_request(
            "POST", f"/_plugins/_ml/models/{self._model_id}/_deploy",
        )
        logger.info("Deploy initiated for model: %s", self._model_id)
        self._wait_for_model_deployed(client)

    def _wait_for_model_deployed(self, client: OpenSearch) -> None:
        cfg = self._config
        for attempt in range(1, cfg.model_poll_max_attempts + 1):
            resp = client.transport.perform_request(
                "GET", f"/_plugins/_ml/models/{self._model_id}",
            )
            state = resp.get("model_state")
            logger.info(
                "Model state (attempt %d/%d): %s",
                attempt,
                cfg.model_poll_max_attempts,
                state,
            )
            if state == "DEPLOYED":
                logger.info("Model deployed successfully.")
                return
            if state in ("DEPLOY_FAILED", "UNDEPLOY_FAILED"):
                raise RuntimeError(f"Model deployment failed: {state}")
            time.sleep(cfg.model_poll_interval_s)

        raise RuntimeError("Model deployment timed out")

    # -- Pipelines -----------------------------------------------------

    def _create_ingest_pipeline(self, client: OpenSearch) -> None:
        name = self._config.ingest_pipeline_name
        try:
            client.transport.perform_request(
                "GET", f"/_ingest/pipeline/{name}",
            )
            logger.info("Ingest pipeline already exists: %s", name)
            return
        except Exception:
            pass

        client.transport.perform_request(
            "PUT",
            f"/_ingest/pipeline/{name}",
            body={
                "description": (
                    "Embeds the 'content' field using Titan Embed v2"
                ),
                "processors": [
                    {
                        "text_embedding": {
                            "model_id": self._model_id,
                            "field_map": {
                                "content": self._config.embedding_field,
                            },
                        },
                    },
                ],
            },
        )
        logger.info("Created ingest pipeline: %s", name)

    def _create_search_pipeline(self, client: OpenSearch) -> None:
        name = self._config.search_pipeline_name
        try:
            client.transport.perform_request(
                "GET", f"/_search/pipeline/{name}",
            )
            logger.info("Search pipeline already exists: %s", name)
            return
        except Exception:
            pass

        client.transport.perform_request(
            "PUT",
            f"/_search/pipeline/{name}",
            body={
                "description": (
                    "Normalisation + weighted combination for hybrid search"
                ),
                "phase_results_processors": [
                    {
                        "normalization-processor": {
                            "normalization": {"technique": "min_max"},
                            "combination": {
                                "technique": "arithmetic_mean",
                                "parameters": {"weights": [0.3, 0.7]},
                            },
                        },
                    },
                ],
            },
        )
        logger.info("Created search pipeline: %s", name)

    # -- Index ---------------------------------------------------------

    def _create_index(self, client: OpenSearch) -> None:
        name = self._config.index_name
        if client.indices.exists(index=name):
            logger.info("Index already exists: %s", name)
            return

        client.indices.create(
            index=name,
            body={
                "settings": {
                    "index": {
                        "knn": True,
                        "default_pipeline": (
                            self._config.ingest_pipeline_name
                        ),
                    },
                },
                "mappings": {
                    "properties": {
                        "filename": {"type": "keyword"},
                        "content": {
                            "type": "text",
                            "analyzer": "standard",
                        },
                        self._config.embedding_field: {
                            "type": "knn_vector",
                            "dimension": self._config.embedding_dim,
                            "method": {
                                "engine": "faiss",
                                "space_type": "l2",
                                "name": "hnsw",
                                "parameters": {},
                            },
                        },
                    },
                },
            },
        )
        logger.info("Created index: %s", name)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self) -> None:
        """Index and retrieve a test document to verify the setup."""
        logger.info("Verification")

        if not self._collection_endpoint:
            self._resolve_collection_endpoint()

        client = self._build_opensearch_client()

        test_doc = {
            "filename": "test-setup-verification.md",
            "content": (
                "Amazon SageMaker is a fully managed machine learning service."
            ),
        }

        resp = client.index(index=self._config.index_name, body=test_doc)
        doc_id = resp["_id"]
        logger.info("Test document indexed: %s", doc_id)

        # Wait for the ingest pipeline to generate the embedding
        time.sleep(3)

        resp = client.get(index=self._config.index_name, id=doc_id)
        source = resp.get("_source", {})
        embedding = source.get(self._config.embedding_field)
        if embedding and len(embedding) > 0:
            logger.info(
                "PASS: Embedding populated (%d dimensions)", len(embedding),
            )
        else:
            logger.warning(
                "WARN: Embedding NOT populated — check ingest pipeline "
                "and model deployment.",
            )

        client.delete(index=self._config.index_name, id=doc_id)
        logger.info("Test document cleaned up.")

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def collection_endpoint(self) -> Optional[str]:
        return self._collection_endpoint


# ---------------------------------------------------------------------------
# Connector Painless scripts (verbatim from the working dashboard setup)
# ---------------------------------------------------------------------------

_CONNECTOR_PRE_PROCESS = """
    StringBuilder builder = new StringBuilder();
    builder.append("{" );
    builder.append("\\"parameters\\":{" );
    builder.append("\\"inputText\\":\\"" );
    builder.append(params.text_docs[0]);
    builder.append("\\"" );
    builder.append("}" );
    builder.append("}" );
    def result = builder.toString();
    return result;
  """

_CONNECTOR_POST_PROCESS = """
    def name = "sentence_embedding";
    def dataType = "FLOAT32";
    if (params.embedding != null) {
      def shape = [params.embedding.length];
      def json = "{" +
        "\\"name\\":\\"" + name + "\\"," +
        "\\"data_type\\":\\"" + dataType + "\\"," +
        "\\"shape\\":" + shape + "," +
        "\\"data\\":" + params.embedding +
        "}";
      return json;
    }
    return "{\\"error\\":\\"No embedding returned\\"}";
  """


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the complete AOSS setup."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        config = AossSetupConfig.from_env()
    except Exception as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    logger.info("Starting AOSS setup for collection: %s", config.collection_name)
    logger.info("Region: %s | Account: %s", config.region, config.account_id)
    logger.info("Caller: %s", config.caller_arn)

    runner = AossSetupRunner(config)

    try:
        runner.setup_iam()
        runner.setup_aoss_infrastructure()
        runner.setup_opensearch_resources()
        runner.verify()
    except Exception as exc:
        logger.error("Setup failed: %s", exc, exc_info=True)
        sys.exit(1)

    print()
    print("=" * 60)
    print("  SETUP COMPLETE")
    print("=" * 60)
    print()
    print("  Add the following to your .env file:")
    print()
    print(f"  OPENSEARCH_ENDPOINT={runner.collection_endpoint}")
    print(f"  OPENSEARCH_INDEX_NAME={config.index_name}")
    print(f"  OPENSEARCH_COLLECTION_NAME={config.collection_name}")
    print()


if __name__ == "__main__":
    main()
