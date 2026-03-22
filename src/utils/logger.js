const LEVELS = { error: 0, warn: 1, info: 2, debug: 3 };
const current = LEVELS[process.env.LOG_LEVEL ?? 'info'] ?? 2;

const fmt = (level, msg, meta) => {
  const ts = new Date().toISOString();
  const base = `[${ts}] [${level.toUpperCase()}] ${msg}`;
  return meta ? `${base} ${JSON.stringify(meta)}` : base;
};

export const logger = {
  error: (msg, meta) => LEVELS.error <= current && console.error(fmt('error', msg, meta)),
  warn:  (msg, meta) => LEVELS.warn  <= current && console.warn(fmt('warn',  msg, meta)),
  info:  (msg, meta) => LEVELS.info  <= current && console.log(fmt('info',   msg, meta)),
  debug: (msg, meta) => LEVELS.debug <= current && console.log(fmt('debug',  msg, meta)),
};
