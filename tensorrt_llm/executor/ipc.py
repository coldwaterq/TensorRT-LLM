import hashlib
import hmac
import importlib
import os
import time
import traceback
from queue import Queue
from typing import Any, Dict, Optional

import msgpack
import zmq
import zmq.asyncio

from tensorrt_llm.logger import logger

from .._utils import nvtx_mark, nvtx_range_debug
from ..llmapi.utils import (ManagedThread, enable_llm_debug, print_colored,
                            print_colored_debug)

_NoValue = object()


class ZeroMqQueue:
    """A Queue-like container for IPC using ZeroMQ."""

    socket_type_str = {
        zmq.PAIR: "PAIR",
        zmq.PULL: "PULL",
        zmq.PUSH: "PUSH",
    }

    # Base serializable types and their handlers
    BASE_SERIALIZABLE_TYPES = {
        "builtins": ["bytes", "Exception", "dict", "list", "tuple", "set"],
        "datetime": ["datetime", "timedelta"],
        "tensorrt_llm.executor.request":
        ["CancellingRequest", "GenerationRequest"],
        "tensorrt_llm.sampling_params": ["SamplingParams"],
        "tensorrt_llm.executor.postproc_worker":
        ["PostprocParams", "PostprocWorker.Input", "PostprocWorker.Output"],
        "tensorrt_llm.serve.postprocess_handlers":
        ["CompletionPostprocArgs", "completion_response_post_processor"],
        "tensorrt_llm.bindings.executor": [
            "Response",
            "Result",
            "FinishReason",
            "KvCacheRetentionConfig",
            "KvCacheRetentionConfig.TokenRangeRetentionConfig",
        ],
        "tensorrt_llm.serve.openai_protocol":
        ["CompletionResponse", "CompletionResponseChoice", "UsageInfo"],
        "torch._utils": ["_rebuild_tensor_v2"],
        "torch.storage": ["_load_from_bytes"],
    }

    def __init__(
        self,
        address: Optional[tuple[str, Optional[bytes]]] = None,
        *,
        socket_type: int = zmq.PAIR,
        is_server: bool,
        is_async: bool = False,
        name: Optional[str] = None,
        additional_serializable_types: Optional[Dict] = None,
        use_hmac_encryption: bool = True,
    ):
        """
        Parameters:
            address (tuple[str, Optional[bytes]], optional): The address (tcp-ip_port, hmac_auth_key) for the IPC. Defaults to None. If hmac_auth_key is None and use_hmac_encryption is False, the queue will not use HMAC encryption.
            is_server (bool): Whether the current process is the server or the client.
            additional_serializable_types (Dict, optional): Additional types to be added to the serializable types.
            use_hmac_encryption (bool): Whether to use HMAC encryption for pickled data. Defaults to True.
        """

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
                "green",
            )
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
        for subpath in attr.split("."):
            obj = getattr(obj, subpath)
        return obj

    def _serialize_obj(self, obj: Any, retMsgPack: bool = True) -> bytes:
        """Safely serialize objects to msgpack using approved types.
        Objects are serialized using the reduce_ex api detailed in https://peps.python.org/pep-0307

        Args:
            obj: The object to serialize
            retMsgPack: Whether to return a msgpack string (True) or Python dict (False)

        Returns:
            Serialized representation of the object as either msgpack string or dict

        Raises:
            PicklingError: If object cannot be serialized
            NotImplementedError: If reduction case not handled
        """

        def format_output(obj: Any):
            """Format output as msgpack string or return as-is based on retMsgPack flag."""
            return msgpack.dumps(obj) if retMsgPack else obj

        # Handle tuples specially since __reduce_ex__ will reduce a tuple into a function
        # call and a tuple resulting in an infinite recursive loop
        if type(obj) is tuple:
            serialized_list = self._serialize_obj(list(obj), retMsgPack=False)
            return format_output({
                "__serial_type__": "tuple",
                "module_name": "builtins",
                "class_name": "tuple",
                "args": serialized_list
            })

        # Try direct msgpack serialization first
        try:
            msgpack.dumps(obj)
            return format_output(obj)
        except TypeError:
            pass

        # __reduce_ex__ will produce function pointers, pickle would write these out
        # as a global instruction, but this is how we handle it
        if callable(obj):
            return format_output({
                "__serial_type__":
                "function",
                "module_name":
                getattr(obj, "__module__", None),
                "class_name":
                getattr(obj, "__qualname__", None),
            })

        # Get reduction method
        reduce = getattr(obj, "__reduce_ex__", _NoValue)
        if reduce is not _NoValue:
            rv = reduce(4)
        else:
            reduce = getattr(obj, "__reduce__", _NoValue)
            if reduce is not _NoValue:
                rv = reduce()
            else:
                raise PicklingError(
                    f"Can't mimic pickle {type(obj).__name__} object: {obj}")

        # Handle the reduced object. __newobj__ is a special case per the spec
        # the rest of the cases are just a function pointer, list of args, and state
        if rv[0].__qualname__ == "__newobj__":
            return self._handle_newobj_reduction(rv, format_output)
        elif len(rv) == 2:
            return format_output({
                "__serial_type__":
                "func_call",
                "module_name":
                getattr(rv[0], "__module__", None),
                "class_name":
                getattr(rv[0], "__qualname__", None),
                "args":
                [self._serialize_obj(arg, retMsgPack=False) for arg in rv[1]],
                "state":
                None,
            })
        elif len(rv) == 3:
            return format_output({
                "__serial_type__":
                "func_call",
                "module_name":
                getattr(rv[0], "__module__", None),
                "class_name":
                getattr(rv[0], "__qualname__", None),
                "args":
                [self._serialize_obj(arg, retMsgPack=False) for arg in rv[1]],
                "state":
                self._serialize_obj(rv[2], retMsgPack=False)
                if rv[2] is not None else None,
            })

        raise NotImplementedError(f"Unhandled reduce case {rv}")

    def _handle_newobj_reduction(self, rv: tuple, format_output: callable):
        """Helper to handle __newobj__ reduction case."""
        return format_output({
            "__serial_type__":
            "newobj",
            "module_name":
            getattr(rv[1][0], "__module__", None),
            "class_name":
            getattr(rv[1][0], "__qualname__", None),
            "args":
            [self._serialize_obj(arg, retMsgPack=False) for arg in rv[1][1:]],
            "state":
            self._serialize_obj(rv[2], retMsgPack=False)
            if rv[2] is not None else None,
            "litems":
            [self._serialize_obj(arg, retMsgPack=False)
             for arg in rv[3]] if rv[3] is not None else None,
            "ditems": {
                k: self._serialize_obj(v, retMsgPack=False)
                for k, v in rv[4]
            } if rv[4] is not None else None,
        })

    def _deserialize_obj(self, data: bytes, isMsgPack: bool = True) -> Any:
        """Helper method to deserialize objects from msgpack using approved types.
        Objects are deserialized from the reduce_ex api object description
        detailed in https://peps.python.org/pep-0307

        Args:
            data (str): The serialized data to deserialize
            isMsgPack (bool): Whether the data is msgpack encoded. Defaults to True.

        Returns:
            Any: The deserialized object

        Raises:
            NotImplementedError: If trying to deserialize an unapproved class
            Exception: If encountering an unhandled serialization type
        """
        if isMsgPack:
            obj = msgpack.loads(data)
        else:
            obj = data

        if not isinstance(obj, dict):
            return obj

        if "__serial_type__" not in obj:
            return obj

        # Prevent arbitrary code execution by only allowing approved imports
        if obj["class_name"] not in self.SERIALIZABLE_TYPES.get(
                obj["module_name"], []):
            raise NotImplementedError(
                f"unapproved class {obj['module_name']} | {obj['class_name']}")

        serial_type = obj["__serial_type__"]

        # Handle basic types
        if serial_type == "tuple":
            return tuple(self._deserialize_obj(obj["args"], isMsgPack=False))

        # Handle function/class types
        module = importlib.import_module(obj["module_name"])
        cls = self._getattr(module, obj["class_name"])

        if serial_type == "function":
            return cls
        elif serial_type == "func_call":
            args = [
                self._deserialize_obj(arg, isMsgPack=False)
                for arg in obj["args"]
            ]
            instance = cls(*args)
            # Restore state if present
            if obj["state"] is not None:
                state = self._deserialize_obj(obj["state"], isMsgPack=False)
                self._restore_state(instance, state)
            return instance
        elif serial_type == "newobj":
            return self._deserialize_newobj(cls, obj)
        else:
            raise Exception(f"unhandled serialization type {serial_type}")

    def _deserialize_newobj(self, cls: type, obj: dict) -> Any:
        """Helper method to deserialize objects using __new__ and state restoration."""
        # Create new instance
        args = [
            self._deserialize_obj(arg, isMsgPack=False) for arg in obj["args"]
        ]
        instance = cls.__new__(cls, *args)

        # Restore state if present
        if obj["state"] is not None:
            state = self._deserialize_obj(obj["state"], isMsgPack=False)
            self._restore_state(instance, state)

        # Restore list items
        if obj["litems"] is not None:
            for item in obj["litems"]:
                instance.append(self._deserialize_obj(item, isMsgPack=False))

        # Restore dict items
        if obj["ditems"] is not None:
            for k, v in obj["ditems"].items():
                instance[k] = self._deserialize_obj(v, isMsgPack=False)

        return instance

    def _restore_state(self, obj: Any, state: Any) -> None:
        """Helper method to restore object state during deserialization."""
        if hasattr(obj, "__setstate__"):
            obj.__setstate__(state)
            return

        slots = None
        if isinstance(state, tuple) and len(state) == 2:
            state, slots = state

        if state:
            obj.__dict__.update(state)

        if slots:
            for k, v in slots.items():
                setattr(obj, k, v)

    def put(self, obj: Any):
        self.setup_lazily()
        with nvtx_range_debug("send", color="blue", category="IPC"):
            if self.use_hmac_encryption:
                # Send pickled data with HMAC appended
                data = self._serialize_obj(obj)
                signed_data = self._sign_data(data)
                self.socket.send(signed_data)
            else:
                # Send data without HMAC
                self.socket.send(self._serialize_obj(obj))

    async def put_async(self, obj: Any):
        self.setup_lazily()
        try:
            if self.use_hmac_encryption:
                # Send pickled data with HMAC appended
                data = self._serialize_obj(obj)
                signed_data = self._sign_data(data)
                await self.socket.send(signed_data)
            else:
                # Send data without HMAC
                await self.socket.send(self._serialize_obj(obj))
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
    """A Queue-like container for IPC with optional message batched."""

    def __init__(
        self,
        address: Optional[tuple[str, Optional[bytes]]] = None,
        *,
        is_server: bool,
        fuse_message=False,
        fuse_size=100000,
        error_queue=None,
        queue_cls=ZeroMqQueue,
        **kwargs,
    ):

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
                "green",
            )

    def close(self):
        self.queue.close()

        if self._send_thread is not None:
            self._send_thread.stop()
            self._send_thread.join()
            self._send_thread = None

        if enable_llm_debug():
            self.print_fuse_stats()
