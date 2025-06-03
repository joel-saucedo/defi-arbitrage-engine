#!/usr/bin/env node

/**
 * Network Latency Performance Testing Suite
 * Tests WebSocket connections, RPC calls, and network optimization
 * Target: Sub-200μs network round-trip times
 */

const WebSocket = require('ws');
const { performance } = require('perf_hooks');
const fs = require('fs').promises;
const cluster = require('cluster');
const os = require('os');

class NetworkLatencyTester {
    constructor() {
        this.results = {
            websocket: [],
            rpc: [],
            tcp: [],
            concurrent: []
        };
        this.testEndpoints = [
            'wss://mainnet.infura.io/ws/v3/YOUR_KEY',  // Replace with actual endpoints
            'wss://eth-mainnet.ws.alchemyapi.io/v2/YOUR_KEY',
            'ws://localhost:8546'
        ];
    }

    // High-precision timing
    hrTime() {
        return performance.now() * 1000; // Convert to microseconds
    }

    // WebSocket latency testing
    async testWebSocketLatency(endpoint, iterations = 1000) {
        const latencies = [];
        
        return new Promise((resolve, reject) => {
            const ws = new WebSocket(endpoint);
            let completed = 0;

            ws.on('open', () => {
                console.log(`Testing WebSocket latency: ${endpoint}`);
                
                const runTest = () => {
                    if (completed >= iterations) {
                        ws.close();
                        resolve(this.calculateStats(latencies));
                        return;
                    }

                    const start = this.hrTime();
                    const requestId = completed + 1;
                    
                    ws.send(JSON.stringify({
                        jsonrpc: "2.0",
                        method: "eth_blockNumber",
                        params: [],
                        id: requestId
                    }));

                    ws.once('message', (data) => {
                        const end = this.hrTime();
                        const latency = end - start;
                        latencies.push(latency);
                        completed++;
                        
                        // Micro-delay to prevent overwhelming
                        setTimeout(runTest, 1);
                    });
                };

                runTest();
            });

            ws.on('error', reject);
            
            // Timeout after 30 seconds
            setTimeout(() => {
                ws.close();
                reject(new Error('WebSocket test timeout'));
            }, 30000);
        });
    }

    // TCP connection latency
    async testTCPLatency(host, port, iterations = 100) {
        const net = require('net');
        const latencies = [];

        for (let i = 0; i < iterations; i++) {
            const start = this.hrTime();
            
            await new Promise((resolve, reject) => {
                const socket = new net.Socket();
                
                socket.connect(port, host, () => {
                    const latency = this.hrTime() - start;
                    latencies.push(latency);
                    socket.destroy();
                    resolve();
                });

                socket.on('error', () => {
                    socket.destroy();
                    resolve(); // Skip failed connections
                });

                setTimeout(() => {
                    socket.destroy();
                    resolve();
                }, 1000);
            });

            // Small delay between connections
            await new Promise(resolve => setTimeout(resolve, 10));
        }

        return this.calculateStats(latencies);
    }

    // Concurrent connection testing
    async testConcurrentConnections(maxConnections = 100) {
        const connections = [];
        const results = [];

        console.log(`Testing ${maxConnections} concurrent connections...`);

        for (let i = 0; i < maxConnections; i++) {
            const promise = this.createConcurrentConnection(i);
            connections.push(promise);
        }

        try {
            const concurrentResults = await Promise.allSettled(connections);
            
            concurrentResults.forEach((result, index) => {
                if (result.status === 'fulfilled') {
                    results.push(result.value);
                }
            });

            return this.calculateStats(results);
        } catch (error) {
            console.error('Concurrent connection test failed:', error);
            return null;
        }
    }

    async createConcurrentConnection(connectionId) {
        return new Promise((resolve, reject) => {
            const ws = new WebSocket('ws://localhost:8546');
            const start = this.hrTime();

            ws.on('open', () => {
                ws.send(JSON.stringify({
                    jsonrpc: "2.0",
                    method: "eth_blockNumber",
                    params: [],
                    id: connectionId
                }));
            });

            ws.on('message', () => {
                const latency = this.hrTime() - start;
                ws.close();
                resolve(latency);
            });

            ws.on('error', () => {
                ws.close();
                reject(new Error(`Connection ${connectionId} failed`));
            });

            setTimeout(() => {
                ws.close();
                reject(new Error(`Connection ${connectionId} timeout`));
            }, 5000);
        });
    }

    // Statistical analysis
    calculateStats(values) {
        if (values.length === 0) return null;

        const sorted = values.sort((a, b) => a - b);
        const sum = values.reduce((a, b) => a + b, 0);
        
        return {
            count: values.length,
            min: Math.min(...values),
            max: Math.max(...values),
            mean: sum / values.length,
            median: sorted[Math.floor(sorted.length / 2)],
            p95: sorted[Math.floor(sorted.length * 0.95)],
            p99: sorted[Math.floor(sorted.length * 0.99)],
            stddev: Math.sqrt(values.reduce((sq, n) => sq + Math.pow(n - (sum / values.length), 2), 0) / values.length)
        };
    }

    // Multi-threaded testing using cluster
    async runMultiThreadedTests() {
        if (cluster.isMaster) {
            console.log('Starting multi-threaded network tests...');
            const numCores = os.cpus().length;
            const workers = [];

            for (let i = 0; i < numCores; i++) {
                const worker = cluster.fork();
                workers.push(worker);
            }

            const results = await Promise.all(workers.map(worker => {
                return new Promise(resolve => {
                    worker.on('message', resolve);
                });
            }));

            workers.forEach(worker => worker.kill());
            return results;
        } else {
            // Worker process
            const tester = new NetworkLatencyTester();
            const result = await tester.testTCPLatency('127.0.0.1', 8545, 50);
            process.send(result);
            process.exit(0);
        }
    }

    // Bandwidth testing
    async testBandwidth() {
        const testSizes = [1024, 4096, 16384, 65536]; // bytes
        const results = [];

        for (const size of testSizes) {
            const data = 'x'.repeat(size);
            const iterations = 100;
            const times = [];

            for (let i = 0; i < iterations; i++) {
                const start = this.hrTime();
                
                // Simulate network operation
                await new Promise(resolve => {
                    const ws = new WebSocket('ws://localhost:8546');
                    
                    ws.on('open', () => {
                        ws.send(JSON.stringify({
                            jsonrpc: "2.0",
                            method: "eth_sendRawTransaction",
                            params: [data],
                            id: i
                        }));
                    });

                    ws.on('message', () => {
                        const time = this.hrTime() - start;
                        times.push(time);
                        ws.close();
                        resolve();
                    });

                    ws.on('error', () => {
                        ws.close();
                        resolve();
                    });
                });
            }

            const stats = this.calculateStats(times);
            if (stats) {
                results.push({
                    size,
                    throughput: (size * 1000000) / stats.mean, // bytes per second
                    latency: stats
                });
            }
        }

        return results;
    }

    // Generate comprehensive report
    async generateReport() {
        const report = {
            timestamp: new Date().toISOString(),
            system: {
                platform: os.platform(),
                arch: os.arch(),
                cpus: os.cpus().length,
                memory: Math.round(os.totalmem() / 1024 / 1024 / 1024) + 'GB'
            },
            tests: {}
        };

        console.log('🚀 Starting comprehensive network latency tests...\n');

        // WebSocket tests
        console.log('📡 Testing WebSocket latency...');
        try {
            report.tests.websocket = await this.testWebSocketLatency('ws://localhost:8546', 500);
            console.log(`   ✓ WebSocket: ${report.tests.websocket.mean.toFixed(2)}μs avg`);
        } catch (error) {
            console.log(`   ✗ WebSocket test failed: ${error.message}`);
        }

        // TCP tests
        console.log('🔌 Testing TCP connection latency...');
        try {
            report.tests.tcp = await this.testTCPLatency('127.0.0.1', 8545, 100);
            console.log(`   ✓ TCP: ${report.tests.tcp.mean.toFixed(2)}μs avg`);
        } catch (error) {
            console.log(`   ✗ TCP test failed: ${error.message}`);
        }

        // Concurrent connections
        console.log('⚡ Testing concurrent connections...');
        try {
            report.tests.concurrent = await this.testConcurrentConnections(50);
            console.log(`   ✓ Concurrent: ${report.tests.concurrent.mean.toFixed(2)}μs avg`);
        } catch (error) {
            console.log(`   ✗ Concurrent test failed: ${error.message}`);
        }

        // Bandwidth tests
        console.log('📊 Testing bandwidth...');
        try {
            report.tests.bandwidth = await this.testBandwidth();
            console.log(`   ✓ Bandwidth tests completed`);
        } catch (error) {
            console.log(`   ✗ Bandwidth test failed: ${error.message}`);
        }

        // Multi-threaded tests
        if (cluster.isMaster) {
            console.log('🧵 Testing multi-threaded performance...');
            try {
                report.tests.multiThreaded = await this.runMultiThreadedTests();
                console.log(`   ✓ Multi-threaded tests completed`);
            } catch (error) {
                console.log(`   ✗ Multi-threaded test failed: ${error.message}`);
            }
        }

        return report;
    }

    // Save results to file
    async saveResults(report) {
        const filename = `network_latency_report_${Date.now()}.json`;
        await fs.writeFile(filename, JSON.stringify(report, null, 2));
        console.log(`\n📄 Report saved to: ${filename}`);
        
        // Also create a summary
        const summary = this.createSummary(report);
        const summaryFile = `network_summary_${Date.now()}.txt`;
        await fs.writeFile(summaryFile, summary);
        console.log(`📋 Summary saved to: ${summaryFile}`);
    }

    createSummary(report) {
        let summary = `NETWORK LATENCY PERFORMANCE REPORT\n`;
        summary += `=====================================\n`;
        summary += `Generated: ${report.timestamp}\n`;
        summary += `System: ${report.system.platform} ${report.system.arch} (${report.system.cpus} cores, ${report.system.memory})\n\n`;

        if (report.tests.websocket) {
            summary += `WebSocket Performance:\n`;
            summary += `  Mean Latency: ${report.tests.websocket.mean.toFixed(2)}μs\n`;
            summary += `  P95 Latency:  ${report.tests.websocket.p95.toFixed(2)}μs\n`;
            summary += `  P99 Latency:  ${report.tests.websocket.p99.toFixed(2)}μs\n`;
            summary += `  Min/Max:      ${report.tests.websocket.min.toFixed(2)}μs / ${report.tests.websocket.max.toFixed(2)}μs\n\n`;
        }

        if (report.tests.tcp) {
            summary += `TCP Connection Performance:\n`;
            summary += `  Mean Latency: ${report.tests.tcp.mean.toFixed(2)}μs\n`;
            summary += `  P95 Latency:  ${report.tests.tcp.p95.toFixed(2)}μs\n`;
            summary += `  Success Rate: ${((report.tests.tcp.count / 100) * 100).toFixed(1)}%\n\n`;
        }

        if (report.tests.concurrent) {
            summary += `Concurrent Connection Performance:\n`;
            summary += `  Mean Latency: ${report.tests.concurrent.mean.toFixed(2)}μs\n`;
            summary += `  Throughput:   ${(1000000 / report.tests.concurrent.mean).toFixed(0)} req/sec\n\n`;
        }

        summary += `PERFORMANCE TARGETS:\n`;
        summary += `  ✓ Sub-200μs WebSocket latency: ${report.tests.websocket && report.tests.websocket.mean < 200 ? 'ACHIEVED' : 'MISSED'}\n`;
        summary += `  ✓ Sub-100μs TCP latency:      ${report.tests.tcp && report.tests.tcp.mean < 100 ? 'ACHIEVED' : 'MISSED'}\n`;
        summary += `  ✓ >1000 req/sec throughput:   ${report.tests.concurrent && (1000000 / report.tests.concurrent.mean) > 1000 ? 'ACHIEVED' : 'MISSED'}\n`;

        return summary;
    }
}

// Main execution
async function main() {
    if (require.main === module) {
        const tester = new NetworkLatencyTester();
        
        try {
            const report = await tester.generateReport();
            await tester.saveResults(report);
            
            console.log('\n🎯 Network latency testing completed!');
            
            // Exit with appropriate code
            const hasFailures = !report.tests.websocket || !report.tests.tcp;
            process.exit(hasFailures ? 1 : 0);
            
        } catch (error) {
            console.error('❌ Network testing failed:', error);
            process.exit(1);
        }
    }
}

main().catch(console.error);

module.exports = { NetworkLatencyTester };
