/**
 * Logger utility using Winston
 */

const winston = require('winston');
const path = require('path');

const logLevel = process.env.LOG_LEVEL || 'info';
const debugMode = process.env.DEBUG_MODE === 'true';

const logger = winston.createLogger({
  level: debugMode ? 'debug' : logLevel,
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'voice-service' },
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.printf(({ timestamp, level, message, ...meta }) => {
          const metaStr = Object.keys(meta).length ? JSON.stringify(meta) : '';
          return `${timestamp} [${level}]: ${message} ${metaStr}`;
        })
      )
    })
  ]
});

const logsDir = path.join(__dirname, '../logs');
try {
  const fs = require('fs');
  if (!fs.existsSync(logsDir)) {
    fs.mkdirSync(logsDir, { recursive: true });
  }
  logger.add(new winston.transports.File({ filename: path.join(logsDir, 'error.log'), level: 'error' }));
  logger.add(new winston.transports.File({ filename: path.join(logsDir, 'combined.log') }));
} catch (_) {}

module.exports = logger;
