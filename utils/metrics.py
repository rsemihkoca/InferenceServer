from prometheus_client import start_http_server, Counter, Histogram

INFERENCE_COUNT = Counter('inference_count', 'Number of inferences performed')
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Latency of inference requests')

def setup_metrics(port):
    start_http_server(port)

def update_inference_count():
    INFERENCE_COUNT.inc()

def update_inference_latency(latency):
    INFERENCE_LATENCY.observe(latency)