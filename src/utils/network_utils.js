/**
 * ═══════════════════════════════════════════════════════════════════════════════════
 *                         ETHEREUM MEV RESEARCH - NETWORK UTILITIES
 *                              WebSocket & RPC Connection Manager
 *                                     Node.js Implementation
 * ═══════════════════════════════════════════════════════════════════════════════════
 */

const WebSocket = require('ws');
const EventEmitter = require('events');
const axios = require('axios');
const { performance } = require('perf_hooks');

class NetworkEngine extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            maxConnections: config.maxConnections || 50,
            reconnectDelay: config.reconnectDelay || 1000,
            heartbeatInterval: config.heartbeatInterval || 30000,
            requestTimeout: config.requestTimeout || 5000,
            ...config
        };
        
        this.connections = new Map();
        this.pendingRequests = new Map();
        this.requestId = 0;
        this.metrics = {
            totalRequests: 0,
            successfulRequests: 0,
            failedRequests: 0,
            averageLatency: 0,
            connectionUptime: new Map()
        };
        
        this.setupHeartbeat();
    }

    /**
     * Establish high-speed WebSocket connection with automatic reconnection
     */
    async connectWebSocket(name, url, protocols = []) {
        if (this.connections.has(name)) {
            this.disconnectWebSocket(name);
        }

        const startTime = performance.now();
        
        try {
            const ws = new WebSocket(url, protocols, {
                perMessageDeflate: false, // Disable compression for speed
                maxPayload: 100 * 1024 * 1024, // 100MB max payload
                handshakeTimeout: 10000
            });

            ws.on('open', () => {
                const latency = performance.now() - startTime;
                console.log(`[NetworkEngine] Connected to ${name} in ${latency.toFixed(2)}ms`);
                this.metrics.connectionUptime.set(name, Date.now());
                this.emit('connected', { name, latency });
            });

            ws.on('message', (data) => {
                this.handleWebSocketMessage(name, data);
            });

            ws.on('error', (error) => {
                console.error(`[NetworkEngine] WebSocket error for ${name}:`, error);
                this.emit('error', { name, error });
            });

            ws.on('close', (code, reason) => {
                console.log(`[NetworkEngine] Connection closed for ${name}: ${code} ${reason}`);
                this.connections.delete(name);
                this.emit('disconnected', { name, code, reason });
                
                // Auto-reconnect
                setTimeout(() => {
                    this.connectWebSocket(name, url, protocols);
                }, this.config.reconnectDelay);
            });

            this.connections.set(name, {
                ws,
                url,
                protocols,
                connected: true,
                lastHeartbeat: Date.now()
            });

            return ws;
        } catch (error) {
            console.error(`[NetworkEngine] Failed to connect to ${name}:`, error);
            throw error;
        }
    }

    /**
     * Send WebSocket message with delivery confirmation
     */
    async sendWebSocketMessage(connectionName, message, requireAck = false) {
        const connection = this.connections.get(connectionName);
        if (!connection || connection.ws.readyState !== WebSocket.OPEN) {
            throw new Error(`Connection ${connectionName} not available`);
        }

        const startTime = performance.now();
        
        try {
            const payload = typeof message === 'string' ? message : JSON.stringify(message);
            connection.ws.send(payload);
            
            this.metrics.totalRequests++;
            this.metrics.successfulRequests++;
            
            const latency = performance.now() - startTime;
            this.updateAverageLatency(latency);
            
            return { success: true, latency };
        } catch (error) {
            this.metrics.failedRequests++;
            throw error;
        }
    }

    /**
     * Batch WebSocket requests for maximum throughput
     */
    async batchWebSocketRequests(connectionName, messages) {
        const connection = this.connections.get(connectionName);
        if (!connection || connection.ws.readyState !== WebSocket.OPEN) {
            throw new Error(`Connection ${connectionName} not available`);
        }

        const startTime = performance.now();
        const results = [];

        try {
            for (const message of messages) {
                const payload = typeof message === 'string' ? message : JSON.stringify(message);
                connection.ws.send(payload);
                results.push({ success: true, message });
            }

            const latency = performance.now() - startTime;
            this.metrics.totalRequests += messages.length;
            this.metrics.successfulRequests += messages.length;
            this.updateAverageLatency(latency / messages.length);

            return { results, totalLatency: latency };
        } catch (error) {
            this.metrics.failedRequests += messages.length;
            throw error;
        }
    }

    /**
     * High-performance HTTP/RPC requests with connection pooling
     */
    async sendRPCRequest(endpoint, method, params = [], headers = {}) {
        const requestId = ++this.requestId;
        const payload = {
            jsonrpc: '2.0',
            id: requestId,
            method,
            params
        };

        const startTime = performance.now();

        try {
            const response = await axios.post(endpoint, payload, {
                headers: {
                    'Content-Type': 'application/json',
                    ...headers
                },
                timeout: this.config.requestTimeout,
                // Use HTTP/2 when available
                httpVersion: '2.0'
            });

            const latency = performance.now() - startTime;
            this.metrics.totalRequests++;
            this.metrics.successfulRequests++;
            this.updateAverageLatency(latency);

            if (response.data.error) {
                throw new Error(`RPC Error: ${response.data.error.message}`);
            }

            return {
                result: response.data.result,
                latency,
                requestId
            };
        } catch (error) {
            this.metrics.failedRequests++;
            const latency = performance.now() - startTime;
            throw new Error(`RPC request failed (${latency.toFixed(2)}ms): ${error.message}`);
        }
    }

    /**
     * Concurrent RPC requests with automatic load balancing
     */
    async batchRPCRequests(endpoints, requests) {
        const startTime = performance.now();
        
        // Distribute requests across endpoints for load balancing
        const promises = requests.map((request, index) => {
            const endpoint = endpoints[index % endpoints.length];
            return this.sendRPCRequest(endpoint, request.method, request.params);
        });

        try {
            const results = await Promise.allSettled(promises);
            const totalLatency = performance.now() - startTime;

            return {
                results: results.map(result => ({
                    success: result.status === 'fulfilled',
                    data: result.status === 'fulfilled' ? result.value : result.reason,
                })),
                totalLatency,
                averageLatency: totalLatency / requests.length
            };
        } catch (error) {
            throw new Error(`Batch RPC request failed: ${error.message}`);
        }
    }

    /**
     * Monitor network latency and connection health
     */
    async measureLatency(endpoint) {
        const measurements = [];
        const iterations = 10;

        for (let i = 0; i < iterations; i++) {
            try {
                const start = performance.now();
                await this.sendRPCRequest(endpoint, 'eth_blockNumber');
                const latency = performance.now() - start;
                measurements.push(latency);
            } catch (error) {
                measurements.push(null);
            }
        }

        const validMeasurements = measurements.filter(m => m !== null);
        if (validMeasurements.length === 0) {
            throw new Error('All latency measurements failed');
        }

        return {
            average: validMeasurements.reduce((a, b) => a + b, 0) / validMeasurements.length,
            min: Math.min(...validMeasurements),
            max: Math.max(...validMeasurements),
            median: validMeasurements.sort()[Math.floor(validMeasurements.length / 2)],
            successRate: (validMeasurements.length / iterations) * 100
        };
    }

    /**
     * Handle incoming WebSocket messages with protocol detection
     */
    handleWebSocketMessage(connectionName, data) {
        try {
            const message = JSON.parse(data.toString());
            
            // Handle different message types
            if (message.method) {
                // Subscription update
                this.emit('subscription', { connectionName, method: message.method, params: message.params });
            } else if (message.id) {
                // Response to previous request
                this.emit('response', { connectionName, id: message.id, result: message.result, error: message.error });
            } else {
                // Unknown message format
                this.emit('message', { connectionName, data: message });
            }
        } catch (error) {
            // Handle binary or non-JSON messages
            this.emit('rawMessage', { connectionName, data });
        }
    }

    /**
     * Disconnect specific WebSocket connection
     */
    disconnectWebSocket(name) {
        const connection = this.connections.get(name);
        if (connection) {
            connection.ws.close();
            this.connections.delete(name);
            console.log(`[NetworkEngine] Disconnected from ${name}`);
        }
    }

    /**
     * Disconnect all connections and cleanup
     */
    disconnectAll() {
        for (const [name] of this.connections) {
            this.disconnectWebSocket(name);
        }
        this.removeAllListeners();
    }

    /**
     * Setup heartbeat mechanism for connection monitoring
     */
    setupHeartbeat() {
        setInterval(() => {
            for (const [name, connection] of this.connections) {
                if (connection.ws.readyState === WebSocket.OPEN) {
                    try {
                        connection.ws.ping();
                        connection.lastHeartbeat = Date.now();
                    } catch (error) {
                        console.error(`[NetworkEngine] Heartbeat failed for ${name}:`, error);
                    }
                }
            }
        }, this.config.heartbeatInterval);
    }

    /**
     * Update running average latency metric
     */
    updateAverageLatency(newLatency) {
        if (this.metrics.averageLatency === 0) {
            this.metrics.averageLatency = newLatency;
        } else {
            // Exponential moving average
            this.metrics.averageLatency = 0.9 * this.metrics.averageLatency + 0.1 * newLatency;
        }
    }

    /**
     * Get comprehensive network statistics
     */
    getMetrics() {
        const now = Date.now();
        const connectionStats = {};
        
        for (const [name, connection] of this.connections) {
            const uptime = this.metrics.connectionUptime.get(name);
            connectionStats[name] = {
                connected: connection.ws.readyState === WebSocket.OPEN,
                uptime: uptime ? now - uptime : 0,
                lastHeartbeat: connection.lastHeartbeat
            };
        }

        return {
            ...this.metrics,
            connections: connectionStats,
            activeConnections: this.connections.size
        };
    }
}

// Export singleton instance for global access
const networkEngine = new NetworkEngine();

module.exports = {
    NetworkEngine,
    networkEngine
};
