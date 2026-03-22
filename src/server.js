import 'dotenv/config';
import express from 'express';
import morgan from 'morgan';
import { initWorkerPool, destroyWorkerPool } from './services/ocr.service.js';
import plateRoutes from './routes/plate.routes.js';
import { logger } from './utils/logger.js';

const PORT = parseInt(process.env.PORT ?? '3001', 10);

const app = express();

// ── Middleware ──────────────────────────────────────────────────────────────
app.use(morgan('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// ── Routes ──────────────────────────────────────────────────────────────────
app.use('/api/plate', plateRoutes);

// 404
app.use((_req, res) => res.status(404).json({ error: 'Not found' }));

// Global error handler
app.use((err, _req, res, _next) => {
  logger.error('Unhandled error', { err: err.message });
  const status = err.status ?? (err.message?.toLowerCase().includes('unsupported') ? 415 : 500);
  res.status(status).json({ success: false, error: err.message });
});

// ── Boot ─────────────────────────────────────────────────────────────────────
async function start() {
  try {
    await initWorkerPool();

    const server = app.listen(PORT, () => {
      logger.info(`licenseplatereader service running`, { port: PORT, env: process.env.NODE_ENV });
      logger.info('Endpoints', {
        detect:   `POST http://localhost:${PORT}/api/plate/detect`,
        validate: `POST http://localhost:${PORT}/api/plate/validate`,
        health:   `GET  http://localhost:${PORT}/api/plate/health`,
      });
    });

    // ── Graceful shutdown ───────────────────────────────────────────────────
    const shutdown = async (signal) => {
      logger.info(`${signal} received — shutting down`);
      server.close(async () => {
        await destroyWorkerPool();
        logger.info('Shutdown complete');
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
