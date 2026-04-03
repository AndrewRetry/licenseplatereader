"""
event_publisher.py — Async RabbitMQ publisher for gantry plate detection events.

Publishes to a topic exchange so downstream services (Arrival Orchestrator,
logging, analytics) can each bind with their own routing-key patterns:

    vehicle.plate.detected   — a plate was read at a gantry
    vehicle.plate.#          — all plate-related events (future: lost, corrected)

Connection uses aio-pika's robust mode: if RabbitMQ restarts or the network
blips, the connection and channel are transparently re-established.

Usage:
    publisher = EventPublisher("amqp://guest:guest@localhost:5672/")
    await publisher.connect()
    await publisher.publish_plate_detected(
        plate_text="SBA1234A",
        confidence=0.94,
        bbox=[120, 340, 380, 420],
        gantry_id="gantry-01",
    )
    await publisher.close()
"""

import json
import logging
from datetime import datetime, timezone

import aio_pika

logger = logging.getLogger(__name__)

EXCHANGE_NAME = "gantry.events"
ROUTING_KEY_PLATE = "vehicle.plate.detected"


class EventPublisher:
    """Publishes plate detection events to RabbitMQ over a topic exchange."""

    def __init__(self, amqp_url: str, exchange_name: str = EXCHANGE_NAME):
        self._amqp_url = amqp_url
        self._exchange_name = exchange_name
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None
        self._exchange: aio_pika.abc.AbstractExchange | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open a robust connection and declare the topic exchange.

        Raises on failure — the caller decides whether to abort or fall
        back to log-only mode.
        """
        logger.info("Connecting to RabbitMQ at %s …", self._amqp_url)
        self._connection = await aio_pika.connect_robust(self._amqp_url)
        self._channel = await self._connection.channel()
        self._exchange = await self._channel.declare_exchange(
            self._exchange_name,
            aio_pika.ExchangeType.TOPIC,
            durable=True,
        )
        logger.info(
            "RabbitMQ publisher ready  exchange=%s  routing=%s",
            self._exchange_name,
            ROUTING_KEY_PLATE,
        )

    async def close(self) -> None:
        """Gracefully shut down the connection."""
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
            logger.info("RabbitMQ connection closed.")

    @property
    def is_connected(self) -> bool:
        return (
            self._connection is not None
            and not self._connection.is_closed
            and self._exchange is not None
        )

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def publish_plate_detected(
        self,
        plate_text: str,
        confidence: float,
        bbox: list[int],
        gantry_id: str,
        frame_timestamp: str | None = None,
    ) -> bool:
        """Publish a ``vehicle.plate.detected`` event.

        Returns True on success, False if the publisher is disconnected
        or the publish fails.
        """
        if not self.is_connected:
            logger.warning("Publisher not connected — dropping event for '%s'", plate_text)
            return False

        payload = {
            "event": ROUTING_KEY_PLATE,
            "timestamp": frame_timestamp or datetime.now(timezone.utc).isoformat(),
            "gantryId": gantry_id,
            "plate": {
                "text": plate_text,
                "confidence": confidence,
                "bbox": bbox,
            },
        }

        message = aio_pika.Message(
            body=json.dumps(payload).encode(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )

        try:
            await self._exchange.publish(message, routing_key=ROUTING_KEY_PLATE)
        except Exception:
            logger.exception("Failed to publish event for '%s'", plate_text)
            return False

        logger.info(
            "Published %s  plate=%s  conf=%.3f  gantry=%s",
            ROUTING_KEY_PLATE, plate_text, confidence, gantry_id,
        )
        return True
