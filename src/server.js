import 'dotenv/config';
import express from 'express';
import morgan from 'morgan';
import path from 'path';
import { fileURLToPath } from 'url';
import { initWorkerPool, destroyWorkerPool } from './services/ocr.service.js';
import plateRoutes from './routes/plate.routes.js';
import { logger } from './utils/logger.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = parseInt(process.env.PORT ?? '3001', 10);

const app = express();

// ── CORS ──────────────────────────────────────────────────────────────────
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin',  process.env.CORS_ORIGIN ?? '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type,Authorization');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

// ── Middleware ─────────────────────────────────────────────────────────────
app.use(morgan('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// ── Static webcam client ───────────────────────────────────────────────────
app.use(express.static(path.resolve(__dirname, '../public')));

// ── API routes ─────────────────────────────────────────────────────────────
app.use('/api/plate', plateRoutes);

// ── 404 ────────────────────────────────────────────────────────────────────
app.use((_req, res) => res.status(404).json({ error: 'Not found' }));

// ── Global error handler ───────────────────────────────────────────────────
app.use((err, _req, res, _next) => {
  logger.error('Unhandled error', { err: err.message });
  const status = err.status
    ?? (err.message?.toLowerCase().includes('unsupported') ? 415 : 500);
  res.status(status).json({ success: false, error: err.message });
});

// ── Boot ───────────────────────────────────────────────────────────────────
async function start() {
  if (!process.env.TESSDATA_PATH) {
    logger.warn(
      'TESSDATA_PATH not set — Tesseract will download eng.traineddata from CDN on first boot. ' +
      'Set TESSDATA_PATH=./tessdata and place eng.traineddata.gz there to avoid this.'
    );
  }

  try {
    await initWorkerPool();

    const server = app.listen(PORT, () => {
      logger.info('licenseplatereader running', { port: PORT, env: process.env.NODE_ENV });
      logger.info('Endpoints', {
        detect:   `POST http://localhost:${PORT}/api/plate/detect`,
        batch:    `POST http://localhost:${PORT}/api/plate/detect/batch`,
        validate: `POST http://localhost:${PORT}/api/plate/validate`,
        health:   `GET  http://localhost:${PORT}/api/plate/health`,
      });
    });

    const shutdown = async (signal) => {
      logger.info(`${signal} — shutting down`);
      server.close(async () => {
        await destroyWorkerPool();
        process.exit(0);
      });
    };
    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT',  () => shutdown('SIGINT'));
  } catch (err) {
    logger.error('Failed to start', { err: err.message });
    process.exit(1);
  }
}

start();