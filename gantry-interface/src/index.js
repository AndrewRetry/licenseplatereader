require("dotenv").config();

const http    = require("http");
const path    = require("path");
const express = require("express");
const { WebSocketServer } = require("ws");
const { createProxyMiddleware } = require("http-proxy-middleware");
const { startConsumer } = require("./amqpConsumer");

// ── Config ────────────────────────────────────────────────────────────────────
const PORT               = parseInt(process.env.PORT ?? "3500");
const AMQP_URL           = process.env.AMQP_URL           ?? "amqp://guest:guest@localhost:5672/";
const AMQP_QUEUE         = process.env.AMQP_QUEUE         ?? "gantry-interface.plate-events";
const DETECT_SERVICE_URL = process.env.DETECT_SERVICE_URL ?? "http://localhost:8081";
const ARRIVAL_API_URL    = process.env.ARRIVAL_API_URL;   // required

if (!ARRIVAL_API_URL) {
  console.error("[boot] ARRIVAL_API_URL is required. Set it in .env");
  process.exit(1);
}

// ── App setup ─────────────────────────────────────────────────────────────────
const app    = express();
const server = http.createServer(app);
const wss    = new WebSocketServer({ server, path: "/ws" });

app.use(express.json());
app.use(express.static(path.join(__dirname, "../public")));

// ── WebSocket broadcast ───────────────────────────────────────────────────────
/**
 * Shared state — kept simple. Not persisted across restarts.
 */
const state = {
  amqp: "disconnected",
  clients: new Set(),
};

function broadcast(type, payload) {
  const message = JSON.stringify({ type, ...payload });
  for (const client of state.clients) {
    if (client.readyState === 1 /* OPEN */) {
      client.send(message);
    }
  }
}

wss.on("connection", (ws) => {
  state.clients.add(ws);
  // Send current AMQP status to the newly connected client
  ws.send(JSON.stringify({ type: "status", amqp: state.amqp }));
  ws.on("close", () => state.clients.delete(ws));
});

// ── Proxy routes → detect_plate_service ──────────────────────────────────────
// These keep the frontend talking to one origin (this service).

const detectProxy = createProxyMiddleware({
  target: DETECT_SERVICE_URL,
  changeOrigin: true,
});

app.post("/stream/start",  detectProxy);
app.post("/stream/stop",   detectProxy);
app.get("/health",         detectProxy);
app.get("/detections",     detectProxy);
app.get("/video",          detectProxy);  // MJPEG camera feed

// ── Status endpoint ───────────────────────────────────────────────────────────
app.get("/interface/status", (_req, res) => {
  res.json({ amqp: state.amqp, arrivalUrl: ARRIVAL_API_URL });
});

// ── AMQP consumer ─────────────────────────────────────────────────────────────
let consumer;

async function boot() {
  consumer = await startConsumer({
    amqpUrl:     AMQP_URL,
    queueName:   AMQP_QUEUE,
    arrivalUrl:  ARRIVAL_API_URL,

    onDetection: (detection) => {
      broadcast("detection", detection);
    },

    onStatus: ({ amqp }) => {
      state.amqp = amqp;
      broadcast("status", { amqp });
    },
  });

  server.listen(PORT, () => {
    console.log(`[boot] Gantry Interface running on http://localhost:${PORT}`);
    console.log(`[boot] Proxying detect_plate_service at ${DETECT_SERVICE_URL}`);
    console.log(`[boot] Arrival API: ${ARRIVAL_API_URL}`);
  });
}

// ── Graceful shutdown ─────────────────────────────────────────────────────────
async function shutdown(signal) {
  console.log(`\n[shutdown] ${signal} received — closing connections...`);
  if (consumer) await consumer.close();
  server.close(() => {
    console.log("[shutdown] HTTP server closed.");
    process.exit(0);
  });
}

process.on("SIGINT",  () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));

boot().catch((err) => {
  console.error("[boot] Fatal error:", err);
  process.exit(1);
});