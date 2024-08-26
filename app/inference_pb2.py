# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: inference.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0finference.proto\x12\tinference\"$\n\x0ePredictRequest\x12\x12\n\nimage_data\x18\x01 \x01(\x0c\"\xfd\x01\n\x0fPredictResponse\x12(\n\ndetections\x18\x01 \x03(\x0b\x32\x14.inference.Detection\x12\x12\n\norig_shape\x18\x02 \x01(\t\x12\r\n\x05\x62oxes\x18\x03 \x01(\t\x12\r\n\x05probs\x18\x04 \x01(\t\x12\x11\n\tkeypoints\x18\x05 \x01(\t\x12\x0b\n\x03obb\x18\x06 \x01(\t\x12\r\n\x05speed\x18\x07 \x01(\t\x12\r\n\x05names\x18\x08 \x01(\t\x12\x13\n\x0bjson_output\x18\t \x01(\t\x12\x12\n\nplot_image\x18\n \x01(\x0c\x12\x16\n\x0everbose_output\x18\x0b \x01(\t\x12\x0f\n\x07summary\x18\x0c \x01(\t\"<\n\tDetection\x12\r\n\x05label\x18\x01 \x01(\t\x12\x12\n\nconfidence\x18\x02 \x01(\x02\x12\x0c\n\x04\x62\x62ox\x18\x03 \x03(\x02\x32V\n\x10InferenceService\x12\x42\n\x07Predict\x12\x19.inference.PredictRequest\x1a\x1a.inference.PredictResponse\"\x00\x62\x06proto3')



_PREDICTREQUEST = DESCRIPTOR.message_types_by_name['PredictRequest']
_PREDICTRESPONSE = DESCRIPTOR.message_types_by_name['PredictResponse']
_DETECTION = DESCRIPTOR.message_types_by_name['Detection']
PredictRequest = _reflection.GeneratedProtocolMessageType('PredictRequest', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTREQUEST,
  '__module__' : 'inference_pb2'
  # @@protoc_insertion_point(class_scope:inference.PredictRequest)
  })
_sym_db.RegisterMessage(PredictRequest)

PredictResponse = _reflection.GeneratedProtocolMessageType('PredictResponse', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTRESPONSE,
  '__module__' : 'inference_pb2'
  # @@protoc_insertion_point(class_scope:inference.PredictResponse)
  })
_sym_db.RegisterMessage(PredictResponse)

Detection = _reflection.GeneratedProtocolMessageType('Detection', (_message.Message,), {
  'DESCRIPTOR' : _DETECTION,
  '__module__' : 'inference_pb2'
  # @@protoc_insertion_point(class_scope:inference.Detection)
  })
_sym_db.RegisterMessage(Detection)

_INFERENCESERVICE = DESCRIPTOR.services_by_name['InferenceService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _PREDICTREQUEST._serialized_start=30
  _PREDICTREQUEST._serialized_end=66
  _PREDICTRESPONSE._serialized_start=69
  _PREDICTRESPONSE._serialized_end=322
  _DETECTION._serialized_start=324
  _DETECTION._serialized_end=384
  _INFERENCESERVICE._serialized_start=386
  _INFERENCESERVICE._serialized_end=472
# @@protoc_insertion_point(module_scope)
