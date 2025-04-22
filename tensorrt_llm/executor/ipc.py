import json
import hashlib
import hmac
import os
import time
import traceback
import importlib
from queue import Queue
from typing import Any, Dict, Optional

import datetime

import zmq
import zmq.asyncio

from ..bindings.executor import KvCacheRetentionConfig

from tensorrt_llm.logger import logger
def get_postproc_params():
    from .postproc_worker import PostprocParams
    return PostprocParams

from ..bindings.executor import Response, Result, FinishReason
from .._utils import nvtx_mark, nvtx_range_debug
from ..llmapi.utils import (ManagedThread, enable_llm_debug, print_colored,
                            print_colored_debug)

# Import postprocessors lazily to avoid circular imports
def get_postprocessor(name: str):
    from tensorrt_llm.serve.postprocess_handlers import (completion_response_post_processor,
                                                       completion_stream_post_processor)
    return {
        "completion_response_post_processor": completion_response_post_processor,
        "completion_stream_post_processor": completion_stream_post_processor,
    }[name]

class ZeroMqQueue:
    ''' A Queue-like container for IPC using ZeroMQ. '''

    socket_type_str = {
        zmq.PAIR: "PAIR",
        zmq.PULL: "PULL",
        zmq.PUSH: "PUSH",
    }

    # Base serializable types and their handlers
    BASE_SERIALIZABLE_TYPES = {
        "builtins.list": {
            "serialize": lambda self, obj: {"__serial_type__": "builtins.list", "objs": [self._serialize_obj(o) for o in obj]},
            "deserialize": lambda self, obj: [self._deserialize_obj(o) for o in obj["objs"]]
        },
        "builtins.bytes": {
            "serialize": lambda self, obj: {"__serial_type__": "builtins.bytes", "data": obj.hex()},
            "deserialize": lambda self, obj: bytes.fromhex(obj["data"])
        },
        "builtins.Exception": {
            "serialize": lambda self, obj: self._basic_serialize_attr(obj, exclude_keys=['add_note']),
            "deserialize": lambda self, obj: self._basic_deserialize_attr(obj)
        },
        "tensorrt_llm.executor.request.GenerationRequest": {
            "serialize": lambda self, obj: self._basic_serialize_attr(obj,exclude_keys=['set_id']),
            "deserialize": lambda self, obj: self._basic_deserialize_attr(obj)
        },
        "tensorrt_llm.executor.request.CancellingRequest": {
            "serialize": lambda self, obj: self._basic_serialize_attr(obj),
            "deserialize": lambda self, obj: self._basic_deserialize_call(obj)
        },
        "tensorrt_llm.executor.postproc_worker.PostprocParams": {
            "serialize": lambda self, obj: {
                "__serial_type__": "tensorrt_llm.executor.postproc_worker.PostprocParams",
                "post_processor": {
                    "__serial_type__": "PostProcessorFunction",
                    "name": obj.post_processor.__name__ if obj.post_processor else None
                },
                "postproc_args": self._serialize_obj(obj.postproc_args)
            },
            "deserialize": lambda self, obj: get_postproc_params()(
                post_processor=get_postprocessor(obj["post_processor"]["name"]),
                postproc_args=self._deserialize_obj(obj["postproc_args"])
            )
        },
        "tensorrt_llm.bindings.executor.Response": {
            "serialize": lambda self, obj: {
                "__serial_type__": "tensorrt_llm.bindings.executor.Response",
                "request_id": obj.request_id,
                "client_id": obj.client_id,
                "error_msg": obj.error_msg if obj.has_error() else "",
                "result": self._serialize_obj(obj.result)
            },
            "deserialize": lambda self, obj: 
                Response(
                request_id=obj["request_id"],
                error_msg=obj["error_msg"],
                client_id=obj["client_id"]) if obj["error_msg"] else Response(
                request_id=obj["request_id"],
                client_id=obj["client_id"],
                result=self._deserialize_obj(obj["result"]))
        },
        "tensorrt_llm.sampling_params.SamplingParams": {
            "serialize": lambda self, obj: self._basic_serialize_attr(obj, exclude_keys=['_greedy_decoding','_get_bad_words','_validate','_setup','_get_stop_words','_get_stop_reasons_and_words','_get_sampling_config','_get_output_config','_get_guided_decoding_params']),
            "deserialize": lambda self, obj: self._basic_deserialize_attr(obj)
        },
        "tensorrt_llm.executor.utils.RequestError": {
            "serialize": lambda self, obj: self._basic_serialize_attr(obj),
            "deserialize": lambda self, obj: self._basic_deserialize_attr(obj)
        },
        "tensorrt_llm.serve.postprocess_handlers.CompletionPostprocArgs": {
            "serialize": lambda self, obj: self._basic_serialize_attr(obj,exclude_keys=['from_request']),
            "deserialize": lambda self, obj: self._basic_deserialize_call(obj)
        },
        "tensorrt_llm.executor.postproc_worker.Input": {
            "serialize": lambda self, obj: self._basic_serialize_attr(obj),
            "deserialize": lambda self, obj: self._basic_deserialize_attr(obj, classDepth=2, class_name="tensorrt_llm.executor.postproc_worker.PostprocWorker.Input")
        },
        "tensorrt_llm.bindings.executor.Result": {
            "serialize": lambda self, obj: self._basic_serialize_attr(obj),
            "deserialize": lambda self, obj: self._basic_deserialize_attr(obj)
        },
        "tensorrt_llm.bindings.executor.FinishReason": {
            "serialize": lambda self, obj: {
                "__serial_type__": "tensorrt_llm.bindings.executor.FinishReason",
                "value": obj.value
            },
            "deserialize": lambda self, obj: 
                FinishReason(obj["value"])
        }
    }

    def __init__(self,
                 address: Optional[tuple[str, Optional[bytes]]] = None,
                 *,
                 socket_type: int = zmq.PAIR,
                 is_server: bool,
                 is_async: bool = False,
                 name: Optional[str] = None,
                 additional_serializable_types: Optional[Dict] = None,
                 use_hmac_encryption: bool = True):
        '''
        Parameters:
            address (tuple[str, Optional[bytes]], optional): The address (tcp-ip_port, hmac_auth_key) for the IPC. Defaults to None. If hmac_auth_key is None and use_hmac_encryption is False, the queue will not use HMAC encryption.
            is_server (bool): Whether the current process is the server or the client.
            additional_serializable_types (Dict, optional): Additional types to be added to the serializable types.
            use_hmac_encryption (bool): Whether to use HMAC encryption for pickled data. Defaults to True.
        '''

        self.socket_type = socket_type
        self.address_endpoint = address[
            0] if address is not None else "tcp://127.0.0.1:*"
        self.is_server = is_server
        self.context = zmq.Context() if not is_async else zmq.asyncio.Context()
        self.poller = None
        self.socket = None

        self._setup_done = False
        self.name = name

        # Initialize SERIALIZABLE_TYPES with base types
        self.SERIALIZABLE_TYPES = self.BASE_SERIALIZABLE_TYPES.copy()
        
        # Add any additional serializable types
        if additional_serializable_types:
            self.SERIALIZABLE_TYPES.update(additional_serializable_types)

        self.socket = self.context.socket(socket_type)

        self.hmac_key = address[1] if address is not None else None
        self.use_hmac_encryption = use_hmac_encryption

        # Check HMAC key condition
        if self.use_hmac_encryption and self.is_server and self.hmac_key is not None:
            raise ValueError(
                "Server should not receive HMAC key when encryption is enabled")
        elif self.use_hmac_encryption and not self.is_server and self.hmac_key is None:
            raise ValueError(
                "Client must receive HMAC key when encryption is enabled")
        elif not self.use_hmac_encryption and self.hmac_key is not None:
            raise ValueError(
                "Server and client should not receive HMAC key when encryption is disabled"
            )

        if (socket_type == zmq.PAIR
                and self.is_server) or socket_type == zmq.PULL:
            self.socket.bind(
                self.address_endpoint
            )  # Binds to the address and occupy a port immediately
            self.address_endpoint = self.socket.getsockopt(
                zmq.LAST_ENDPOINT).decode()
            print_colored_debug(
                f"Server [{name}] bound to {self.address_endpoint} in {self.socket_type_str[socket_type]}\n",
                "green")

            if self.use_hmac_encryption:
                # Initialize HMAC key for pickle encryption
                logger.info(f"Generating a new HMAC key for server {self.name}")
                self.hmac_key = os.urandom(32)

            self.address = (self.address_endpoint, self.hmac_key)

    def setup_lazily(self):
        if self._setup_done:
            return
        self._setup_done = True

        if not self.is_server:
            print_colored_debug(
                f"Client [{self.name}] connecting to {self.address_endpoint} in {self.socket_type_str[self.socket_type]}\n",
                "green")
            self.socket.connect(self.address_endpoint)

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

    def poll(self, timeout: int) -> bool:
        """
        Parameters:
            timeout (int): Timeout in seconds
        """
        self.setup_lazily()

        events = dict(self.poller.poll(timeout=timeout * 1000))
        if self.socket in events and events[self.socket] == zmq.POLLIN:
            return True
        else:
            return False

    def _basic_serialize_dict(self, obj: Any) -> str:
        obj_dict = obj.__dict__
        for key, value in obj_dict.items():
            obj_dict[key] = self._serialize_obj(value)
        
        obj_dict["__serial_type__"] = obj.__class__.__module__ + "." + obj.__class__.__name__
        return obj_dict

    def _basic_deserialize_call(self, obj_dict: Dict[str, str]) -> str:
        # Get the class from the module path
        module_path, class_name = obj_dict["__serial_type__"].rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        
        del(obj_dict["__serial_type__"])
        for key, value in obj_dict.items():
            obj_dict[key] = self._deserialize_obj(value)

        instance = cls(**obj_dict)
                
        return instance

    def _basic_serialize_attr(self, obj: Any, exclude_keys: list = None) -> str:
        obj_dict = {}
        # Get all public attributes using dir()
        print(obj.__class__.__name__)
        for attr in dir(obj):
            # Skip private/special attributes and methods
            if not attr.startswith('__'):
                # Skip excluded keys
                if exclude_keys and attr in exclude_keys:
                    continue
                value = getattr(obj, attr)
                # Only serialize if not a method/function
                obj_dict[attr] = self._serialize_obj(value)
        
        obj_dict["__serial_type__"] = obj.__class__.__module__ + "." + obj.__class__.__name__
        return obj_dict

    def _basic_deserialize_attr(self, obj_dict: Dict[str, str], classDepth: int = 1, class_name: str = None) -> Any:
        # Get the class from the module path
        if class_name is None:
            class_name = obj_dict["__serial_type__"]
        class_names = class_name.rsplit(".", classDepth)
        module_path = class_names[0]
        class_names = class_names[1:]

        module = importlib.import_module(module_path)
        cls = getattr(module, class_names[0])
        for i in range(1,len(class_names)):
            cls = getattr(cls, class_names[i])
        # Create a new instance
        instance = cls.__new__(cls)
        
        # Deserialize each field
        for key, value in obj_dict.items():
            if key != "__serial_type__":
                setattr(instance, key, self._deserialize_obj(value))
                
        return instance

    def _serialize_obj(self, obj: Any) -> str:
        """Helper method to safely serialize objects to JSON using approved types"""
        if obj.__class__.__module__ + "." + obj.__class__.__name__ in self.SERIALIZABLE_TYPES.keys():
            type_handler = self.SERIALIZABLE_TYPES[obj.__class__.__module__ + "." + obj.__class__.__name__]
            try:
                serialized_obj = type_handler["serialize"](self,obj)
                return json.dumps(serialized_obj)
            except Exception as e:
                logger.error(f"Error serializing object: {e}")
                logger.error(f"Object: {obj}")
                raise e
                
        # If object is not in approved list, try basic JSON serialization
        try:
            return json.dumps(obj)
        except TypeError as e:
            logger.error(f"Unserializable object: {obj}")
            raise TypeError(f"Object {obj.__class__.__module__}.{obj.__class__.__name__} is not in approved serializable types") from e

    def _deserialize_obj(self, data: str) -> Any:
        """Helper method to deserialize objects from JSON using approved types"""
        if type(data) == bytes:
            data = data.decode('utf-8')
        obj = json.loads(data)
        if isinstance(obj, dict):
            if "__serial_type__" in obj.keys():
                return self.SERIALIZABLE_TYPES[obj["__serial_type__"]]["deserialize"](self,obj)
        return obj

    def put(self, obj: Any):
        self.setup_lazily()
        with nvtx_range_debug("send", color="blue", category="IPC"):
            if self.use_hmac_encryption:
                # Send pickled data with HMAC appended
                data = self._serialize_obj(obj).encode('utf-8')
                signed_data = self._sign_data(data)
                self.socket.send(signed_data)
            else:
                # Send data without HMAC
                self.socket.send(self._serialize_obj(obj).encode('utf-8'))

    async def put_async(self, obj: Any):
        self.setup_lazily()
        try:
            if self.use_hmac_encryption:
                # Send pickled data with HMAC appended
                data = self._serialize_obj(obj).encode('utf-8')
                signed_data = self._sign_data(data)
                await self.socket.send(signed_data)
            else:
                # Send data without HMAC
                await self.socket.send(self._serialize_obj(obj).encode('utf-8'))
        except TypeError as e:
            logger.error(f"Cannot pickle {obj}")
            raise e
        except Exception as e:
            logger.error(f"Error sending object: {e}")
            logger.error(traceback.format_exc())
            raise e

        nvtx_mark("ipc.send", color="blue", category="IPC")

    def get(self) -> Any:
        self.setup_lazily()

        if self.use_hmac_encryption:
            # Receive signed data with HMAC
            signed_data = self.socket.recv()

            # Split data and HMAC
            data = signed_data[:-32]
            actual_hmac = signed_data[-32:]

            # Verify HMAC
            if not self._verify_hmac(data, actual_hmac):
                raise RuntimeError("HMAC verification failed")

            obj = self._deserialize_obj(data)
        else:
            # Receive data without HMAC
            obj = self._deserialize_obj(self.socket.recv())
        return obj

    async def get_async(self) -> Any:
        self.setup_lazily()

        if self.use_hmac_encryption:
            # Receive signed data with HMAC
            signed_data = await self.socket.recv()

            # Split data and HMAC
            data = signed_data[:-32]
            actual_hmac = signed_data[-32:]

            # Verify HMAC
            if not self._verify_hmac(data, actual_hmac):
                raise RuntimeError("HMAC verification failed")

            obj = self._deserialize_obj(data)  
        else:
            # Receive data without HMAC
            obj = self._deserialize_obj(await self.socket.recv())
        return obj

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None

    def _verify_hmac(self, data: bytes, actual_hmac: bytes) -> bool:
        """Verify the HMAC of received pickle data."""
        expected_hmac = hmac.new(self.hmac_key, data, hashlib.sha256).digest()
        return hmac.compare_digest(expected_hmac, actual_hmac)

    def _sign_data(self, data_before_encoding: bytes) -> bytes:
        """Generate HMAC for data."""
        hmac_signature = hmac.new(self.hmac_key, data_before_encoding,
                                  hashlib.sha256).digest()
        return data_before_encoding + hmac_signature

    def __del__(self):
        self.close()


IpcQueue = ZeroMqQueue


class FusedIpcQueue:
    ''' A Queue-like container for IPC with optional message batched. '''

    def __init__(self,
                 address: Optional[tuple[str, Optional[bytes]]] = None,
                 *,
                 is_server: bool,
                 fuse_message=False,
                 fuse_size=100000,
                 error_queue=None,
                 queue_cls=ZeroMqQueue,
                 **kwargs):

        self.queue = queue_cls(address=address, is_server=is_server, **kwargs)
        self.fuse_message = fuse_message
        self.error_queue = error_queue
        self.fuse_size = fuse_size
        self._message_counter = 0
        self._obj_counter = 0
        self._send_thread = None
        self.sending_queue = Queue() if fuse_message else None

    def setup_sender(self):
        if not self.fuse_message or self._send_thread is not None:
            return

        def send_task():
            while True:
                qsize = self.sending_queue.qsize()
                if qsize > 0:
                    qsize = min(self.fuse_size, qsize)
                    self._obj_counter += qsize
                    message = [
                        self.sending_queue.get_nowait() for _ in range(qsize)
                    ]
                    self.queue.put(message)
                    self._message_counter += 1
                else:
                    time.sleep(0.001)

        self._send_thread = ManagedThread(send_task,
                                          name="fused_send_thread",
                                          error_queue=self.error_queue)
        self._send_thread.start()

    def put(self, obj: Any):
        self.setup_sender()
        if self.fuse_message:
            self.sending_queue.put_nowait(obj)
        else:
            batch = obj if isinstance(obj, list) else [obj]
            self.queue.put(batch)

    def get(self) -> Any:
        return self.queue.get()

    @property
    def address(self) -> tuple[str, Optional[bytes]]:
        return self.queue.address

    def __del__(self):
        self.close()

    def print_fuse_stats(self):
        if self._message_counter > 0:
            print_colored(
                f"IPCQueue: {self._message_counter} messages, {self._obj_counter} objects sent, average: {self._obj_counter/self._message_counter}.\n",
                "green")

    def close(self):
        self.queue.close()

        if self._send_thread is not None:
            self._send_thread.stop()
            self._send_thread.join()
            self._send_thread = None

        if enable_llm_debug():
            self.print_fuse_stats()
