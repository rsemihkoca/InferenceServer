from prometheus_client import make_asgi_app, Summary

INFERENCE_TIME = Summary('inference_time_seconds', 'Time spent processing inference')

def setup_metrics(app):
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)