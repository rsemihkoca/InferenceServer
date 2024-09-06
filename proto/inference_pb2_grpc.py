# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import inference_pb2 as inference__pb2


class InferenceServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Predict = channel.unary_unary(
                '/inference.InferenceService/Predict',
                request_serializer=inference__pb2.PredictRequest.SerializeToString,
                response_deserializer=inference__pb2.PredictResponse.FromString,
                )
        self.BatchPredict = channel.unary_unary(
                '/inference.InferenceService/BatchPredict',
                request_serializer=inference__pb2.BatchPredictRequest.SerializeToString,
                response_deserializer=inference__pb2.BatchPredictResponse.FromString,
                )
        self.TestPredict = channel.unary_unary(
                '/inference.InferenceService/TestPredict',
                request_serializer=inference__pb2.TestPredictRequest.SerializeToString,
                response_deserializer=inference__pb2.TestPredictResponse.FromString,
                )


class InferenceServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Predict(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def BatchPredict(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def TestPredict(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InferenceServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Predict': grpc.unary_unary_rpc_method_handler(
                    servicer.Predict,
                    request_deserializer=inference__pb2.PredictRequest.FromString,
                    response_serializer=inference__pb2.PredictResponse.SerializeToString,
            ),
            'BatchPredict': grpc.unary_unary_rpc_method_handler(
                    servicer.BatchPredict,
                    request_deserializer=inference__pb2.BatchPredictRequest.FromString,
                    response_serializer=inference__pb2.BatchPredictResponse.SerializeToString,
            ),
            'TestPredict': grpc.unary_unary_rpc_method_handler(
                    servicer.TestPredict,
                    request_deserializer=inference__pb2.TestPredictRequest.FromString,
                    response_serializer=inference__pb2.TestPredictResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'inference.InferenceService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class InferenceService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Predict(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/inference.InferenceService/Predict',
            inference__pb2.PredictRequest.SerializeToString,
            inference__pb2.PredictResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def BatchPredict(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/inference.InferenceService/BatchPredict',
            inference__pb2.BatchPredictRequest.SerializeToString,
            inference__pb2.BatchPredictResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def TestPredict(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/inference.InferenceService/TestPredict',
            inference__pb2.TestPredictRequest.SerializeToString,
            inference__pb2.TestPredictResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
