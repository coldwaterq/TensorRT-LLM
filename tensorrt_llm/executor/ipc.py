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

_NoValue = object()

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
        "builtins": ["bytes","Exception","dict","list","tuple","set"],
        "datetime": ["datetime", "timedelta"],
        "tensorrt_llm.executor.request": ["CancellingRequest","GenerationRequest"],
        "tensorrt_llm.sampling_params":["SamplingParams"],
        "tensorrt_llm.executor.postproc_worker":["PostprocParams","PostprocWorker.Input","PostprocWorker.Output"],
        "tensorrt_llm.serve.postprocess_handlers":["CompletionPostprocArgs","completion_response_post_processor"],
        "tensorrt_llm.bindings.executor":["Response","Result","FinishReason","KvCacheRetentionConfig","KvCacheRetentionConfig.TokenRangeRetentionConfig"],
        "tensorrt_llm.serve.openai_protocol":["CompletionResponse","CompletionResponseChoice","UsageInfo"],
        "torch._utils":["_rebuild_tensor_v2"],
        "torch.storage":["_load_from_bytes"],
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

    def _getattr(self, obj: Any, attr: str) -> Any:
        for subpath in attr.split('.'):
            obj = getattr(obj, subpath)
        return obj

    def _serialize_obj(self, obj: Any, retJson: bool = True) -> str:
        """Safely serialize objects to JSON using approved types.
        
        Args:
            obj: The object to serialize
            retJson: Whether to return a JSON string (True) or Python dict (False)
            
        Returns:
            Serialized representation of the object as either JSON string or dict
            
        Raises:
            PicklingError: If object cannot be serialized
            NotImplementedError: If reduction case not handled
        """
        def format_output(obj: Any):
            """Format output as JSON string or return as-is based on retJson flag."""
            return json.dumps(obj) if retJson else obj

        # Handle tuples specially since JSON doesn't support them
        if type(obj) is tuple:
            serialized_list = self._serialize_obj(list(obj), retJson=False)
            return format_output({
                "__serial_type__": "tuple",
                "module_name": "builtins", 
                "class_name": "tuple",
                "args": serialized_list
            })

        # Try direct JSON serialization first
        try:
            json.dumps(obj)
            return format_output(obj)
        except TypeError:
            pass

        # Handle special cases
        if type(obj) is bytes:
            return format_output({
                "__serial_type__": "bytes",
                "module_name": "builtins",
                "class_name": "bytes", 
                "args": [obj.hex()]
            })

        if callable(obj):
            return format_output({
                "__serial_type__": "function",
                "module_name": getattr(obj, "__module__", None),
                "class_name": getattr(obj, "__qualname__", None)
            })

        # Handle classes with custom metaclass
        if issubclass(type(obj), type):
            return format_output(obj)

        # Get reduction method
        reduce = getattr(obj, "__reduce_ex__", _NoValue)
        if reduce is not _NoValue:
            rv = reduce(4)
        else:
            reduce = getattr(obj, "__reduce__", _NoValue)
            if reduce is not _NoValue:
                rv = reduce()
            else:
                raise PicklingError(f"Can't pickle {type(obj).__name__} object: {obj}")

        # Handle different reduction cases
        if rv[0].__qualname__ == "__newobj__":
            return self._handle_newobj_reduction(rv, format_output)
        elif rv[0].__qualname__ == "Exception":
            return format_output({
                "__serial_type__": "exception",
                "module_name": "builtins",
                "class_name": "Exception",
                "args": [rv[1][0]]
            })
        elif rv[0].__qualname__ == "set":
            return format_output({
                "__serial_type__": "set",
                "module_name": "builtins",
                "class_name": "set",
                "args": self._serialize_obj(rv[1][0], retJson=False)
            })
        elif len(rv) == 2:
            return format_output({
                "__serial_type__": "func_call",
                "module_name": getattr(rv[0], "__module__", None),
                "class_name": getattr(rv[0], "__qualname__", None),
                "args": [self._serialize_obj(arg, retJson=False) for arg in rv[1]]
            })

        raise NotImplementedError(f"Unhandled reduce case {rv}")

    def _handle_newobj_reduction(self, rv: tuple, format_output: callable):
        """Helper to handle __newobj__ reduction case."""
        return format_output({
            "__serial_type__": "newobj",
            "module_name": getattr(rv[1][0], "__module__", None),
            "class_name": getattr(rv[1][0], "__qualname__", None),
            "args": [self._serialize_obj(arg, retJson=False) for arg in rv[1][1:]],
            "state": self._serialize_obj(rv[2], retJson=False) if rv[2] is not None else None,
            "litems": [self._serialize_obj(arg, retJson=False) for arg in rv[3]] if rv[3] is not None else None,
            "ditems": {k: self._serialize_obj(v, retJson=False) for k,v in rv[4]} if rv[4] is not None else None
        })

    def _deserialize_obj(self, data: str, isJson: bool = True) -> Any:
        """Helper method to deserialize objects from JSON using approved types"""
        
        if isJson:
            if type(data) == bytes:
                data = data.decode('utf-8')
            obj = json.loads(data)
        else:
            obj = data
        
        if isinstance(obj, dict):
            if "__serial_type__" in obj:
                if obj["class_name"] not in self.SERIALIZABLE_TYPES.get(obj['module_name'],[]):
                    raise NotImplementedError(f"unapproved class {obj['module_name']} | {obj['class_name']}")
                elif obj["__serial_type__"] == "bytes":
                    obj = bytes.fromhex(obj["args"][0])
                elif obj["__serial_type__"] == "tuple":
                    obj = tuple(self._deserialize_obj(obj["args"], isJson=False))
                elif obj["__serial_type__"] == "function":
                    module = importlib.import_module(obj["module_name"])
                    cls = self._getattr(module, obj["class_name"])
                    obj = cls
                elif obj["__serial_type__"] == "exception":
                    obj = Exception(obj["args"][0])
                elif obj["__serial_type__"] == "func_call":
                    module = importlib.import_module(obj["module_name"])
                    cls = self._getattr(module, obj["class_name"])
                    obj = cls(*[self._deserialize_obj(arg, isJson=False) for arg in obj["args"]])
                elif obj["__serial_type__"] == "newobj":
                    # Handle newobj case from _serialize_obj
                    module = importlib.import_module(obj["module_name"])
                    cls = self._getattr(module, obj["class_name"])
                    args = [self._deserialize_obj(arg, isJson=False) for arg in obj["args"]]
                    init_obj =  cls.__new__(cls, *args)
                    if obj["state"] is not None:
                        state = self._deserialize_obj(obj["state"], isJson=False)
                        if hasattr(init_obj, "__setstate__"):
                            init_obj.__setstate__(state)
                        else:
                            slots = None
                            if isinstance(state, tuple) and len(state) == 2:
                                state, slots = state
                            if state:
                                for k, v in state.items():
                                    init_obj.__dict__[k] = v
                            if slots:
                                for k, v in slots.items():
                                    setattr(init_obj, k, v)
                        
                    if obj["litems"] is not None:
                        for v in obj["litems"]:
                            init_obj.append(self._deserialize_obj(v, isJson=False))
                    if obj["ditems"] is not None:
                        for k,v in obj["ditems"].items():
                            init_obj[k] = self._deserialize_obj(v, isJson=False)
                    obj = init_obj  
                elif obj["__serial_type__"] == "set":
                    obj = set(self._deserialize_obj(obj["args"], isJson=False))
                else:
                    # Handle other serialization types
                    raise Exception(f"unhandled serialization type {obj['__serial_type__']}")
        
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
