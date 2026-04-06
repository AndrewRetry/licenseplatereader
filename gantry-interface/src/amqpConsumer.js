const amqp = require("amqplib");
const axios = require("axios");

const EXCHANGE      = "gantry.events";
const ROUTING_KEY   = "vehicle.plate.detected";
const RECONNECT_MS  = 5000;

/**
 * Subscribes to vehicle.plate.detected on the gantry.events exchange.
 * On each event:
 *   1. Calls POST /arrival with { plate: string }
 *   2. Broadcasts the detection to all connected WebSocket clients
 *
 * @param {Object} opts
 * @param {string}   opts.amqpUrl      - RabbitMQ connection string
 * @param {string}   opts.queueName    - Durable queue name for this consumer
 * @param {string}   opts.arrivalUrl   - Full URL for POST /arrival
 * @param {Function} opts.onDetection  - Callback(detectionPayload) to broadcast via WS
 * @param {Function} opts.onStatus     - Callback({ amqp: 'connected'|'disconnected' })
 */
async function startConsumer({ amqpUrl, queueName, arrivalUrl, onDetection, onStatus }) {
  let conn, channel;

  async function connect() {
    try {
      conn = await amqp.connect(amqpUrl);
      channel = await conn.createChannel();

      await channel.assertExchange(EXCHANGE, "topic", { durable: true });
      await channel.assertQueue(queueName, { durable: true });
      await channel.bindQueue(queueName, EXCHANGE, ROUTING_KEY);
      await channel.prefetch(5);

      onStatus({ amqp: "connected" });
      console.log(`[amqp] Connected. Listening on exchange=${EXCHANGE} key=${ROUTING_KEY}`);

      channel.consume(queueName, async (msg) => {
        if (!msg) return;

        let event;
        try {
          event = JSON.parse(msg.content.toString());
        } catch {
          console.error("[amqp] Malformed message — discarding.");
          return channel.nack(msg, false, false);
        }

        const plateText = event?.plate?.text;
        if (!plateText) {
          console.warn("[amqp] Missing plate.text in event — discarding.");
          return channel.nack(msg, false, false);
        }

        // 1. Call arrival API
        try {
          await axios.post(arrivalUrl, { license_plate: plateText });
          console.log(`[amqp] Arrival notified for plate=${plateText}`);
        } catch (err) {
          console.error(`[amqp] POST /arrival failed for plate=${plateText}: ${err.message}`);
          // Nack without requeue — arrival failures should not block the queue
          return channel.nack(msg, false, false);
        }

        // 2. Broadcast to dashboard WebSocket clients
        onDetection({
          plate: plateText,
          confidence: event?.plate?.confidence ?? null,
          gantryId: event?.gantryId ?? null,
          timestamp: event?.timestamp ?? new Date().toISOString(),
        });

        channel.ack(msg);
      });

      conn.on("error", (err) => console.error("[amqp] Connection error:", err.message));
      conn.on("close", () => {
        onStatus({ amqp: "disconnected" });
        console.warn(`[amqp] Connection closed — retrying in ${RECONNECT_MS}ms`);
        setTimeout(connect, RECONNECT_MS);
      });
    } catch (err) {
      onStatus({ amqp: "disconnected" });
      console.error(`[amqp] Failed to connect: ${err.message} — retrying in ${RECONNECT_MS}ms`);
      setTimeout(connect, RECONNECT_MS);
    }
  }

  await connect();

  return {
    close: async () => {
      try {
        if (channel) await channel.close();
        if (conn) await conn.close();
      } catch {}
    },
  };
}

module.exports = { startConsumer };