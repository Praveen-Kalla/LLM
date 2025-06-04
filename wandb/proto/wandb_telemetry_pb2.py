import google.protobuf

protobuf_version = google.protobuf.__version__[0]

if protobuf_version == "3":
    from wandb.proto.v3.wandb_telemetry_pb2 import *
elif protobuf_version == "4":
    from wandb.proto.v4.wandb_telemetry_pb2 import *
else:
    # Protobuf 5 introduced a new package name which we support via v5
    # Protocol Buffers 6.x remains backwards compatible with the v5 API,
    # so fall back to the v5 generated module for any newer versions.
    from wandb.proto.v5.wandb_telemetry_pb2 import *
