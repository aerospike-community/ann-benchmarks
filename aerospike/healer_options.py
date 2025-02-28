import asyncio
import time
import logging

from typing import Union, Optional
from logging import _nameToLevel as LogLevels

from baseaerospike import BaseAerospike
from aerospike_vector_search import types as vectorTypes
from aerospike_vector_search.aio import Client as vectorASyncClient

class HealerOptions(object):

    DISABLE:int = 0
    NOW:int = -1
    SETRT:int = -2
    NOOPT = None

    @staticmethod
    def seconds_to_cron(seconds:int) -> str:
        if seconds >= 3600:
            return "0 0/{interval} * * * ?".format(interval = seconds // 60)
        elif seconds >= 60:
            return "0/{interval} * * * * ?".format(interval=seconds // 60)
        else:
            return "0/{seconds} * * * * ?".format(seconds=seconds)

    @staticmethod
    def _determineschedulerstr(schedulerSecs:Optional[int]) -> Optional[str]:

        if schedulerSecs is None:
            return None
        elif schedulerSecs == HealerOptions.DISABLE:
            return "0 0 0 1 1 ? 2099"
        elif schedulerSecs == HealerOptions.NOW:
            return "* * * * * ?"
        elif schedulerSecs < 0:
            return None
        else:
            return HealerOptions.seconds_to_cron(schedulerSecs)

    @staticmethod
    def Determine_State(schedulerSecs:Optional[int]) -> str:
        scheduleParams:str = 'N/A'
        if schedulerSecs is None:
            scheduleParams:str = 'NoOp'
        elif schedulerSecs == HealerOptions.DISABLE:
            scheduleParams:str = 'Disable'
        elif schedulerSecs == HealerOptions.NOW:
            scheduleParams:str = 'Now'
        elif schedulerSecs == HealerOptions.SETRT:
            scheduleParams:str = 'SetRT'
        elif schedulerSecs > 0:
            scheduleParams:str = f'{schedulerSecs} secs'

        return scheduleParams

    def __init__(self, schedulerSecs:Optional[int],
                        hdfInstance: BaseAerospike,
                        asyncclient: vectorASyncClient,
                        logger:logging.Logger,
                        delaytimesec:int = 1):
        '''
        schedulerSecs -- 0  - disable healer,
                            -1 - Now
                            -2 - Save and restore (can change Params)
                            >0 - Number of seconds interval
                            None - No operation
        '''
        self._vector_idxParams: Optional[vectorTypes.IndexDefinition] = None
        self._vector_asyncClient: vectorASyncClient = asyncclient
        self._hdfInstance = hdfInstance
        self._logger:logging.Logger = logger
        self._schedulerSecs:Optional[int] = schedulerSecs
        self._vector_idxhealerScheduler:Optional[str] = HealerOptions._determineschedulerstr(schedulerSecs)
        self._delaytimesecs = delaytimesec
        self._logger.info(f"HealerOptions {self}")

    async def __aenter__(self):
        if self._schedulerSecs == -2:
            await self.SaveIdxParams()
        elif self._vector_idxhealerScheduler is not None:
            await self.SaveIdxParams()
            await self.SetIdxParams()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._logger.exception(f"HealerOptions: error detected on exit.")
        await self.RestoreIdxParams()

    def IsDisabled(self) -> bool:
        return self._schedulerSecs == HealerOptions.DISABLE

    def IsRunNow(self) -> bool:
        return self._schedulerSecs == HealerOptions.NOW

    def IsSetRT(self) -> bool:
        return self._schedulerSecs == HealerOptions.SETRT

    def IsNoOperation(self) -> bool:
        return self._schedulerSecs is None

    async def SaveIdxParams(self) -> Optional[vectorTypes.IndexDefinition]:
        from aerospike_vector_search.shared.proto_generated.types_pb2_grpc import grpc  as vectorResultCodes

        if self._logger.level == logging.DEBUG:
                self._logger.debug(f"HealerOptions.SaveIdxParams {self}")

        try:
            self._vector_idxParams = await self._vector_asyncClient.index_get(namespace=self._hdfInstance._namespace,
                                                                                name=self._hdfInstance._idx_name,
                                                                                timeout=4)

            self._logger.info(f"HealerOptions.SaveIdxParams {self}")

        except vectorTypes.AVSServerError as avse:
            self._vector_idxParams = None
            if (avse.rpc_error.code() == vectorResultCodes.StatusCode.NOT_FOUND):
                self._logger.warning(f"HealerOptions.SaveIdxParams Index not found, ignoring for {self}")
            else:
                raise
        except Exception as e:
            self._vector_idxParams = None
            self._logger.exception(f"HealerOptions.SaveIdxParams failed ns={self._hdfInstance._namespace}, name={self._hdfInstance._idx_name}")
        return self._vector_idxParams

    async def RestoreIdxParams(self) -> Optional[vectorTypes.IndexDefinition]:
        if self._vector_idxParams is None:
            self._logger.info(f"HealerOptions.RestoreIdxParams Nothing to Restore {self}")
            return None

        self._logger.info(f"HealerOptions.RestoreIdxParams {self}")

        try:
            await self._vector_asyncClient.index_update(namespace=self._hdfInstance._namespace,
                                                        name=self._hdfInstance._idx_name,
                                                        hnsw_update_params=vectorTypes.HnswIndexUpdate(healer_params=self._vector_idxParams.hnsw_params.healer_params))
            await asyncio.sleep(self._delaytimesecs) #need to wait to take effect

            if self._logger.level == logging.DEBUG:
                params = await self._vector_asyncClient.index_get(namespace=self._hdfInstance._namespace,
                                                                    name=self._hdfInstance._idx_name,
                                                                    timeout=4)
                self._logger.debug(f"HealerOptions.RestoreIdxParams Restored to '{'None' if params is None else params.hnsw_params.healer_params.schedule}'")

        except Exception as e:
            self._logger.exception(f"HealerOptions.RestoreIdxParams failed ns={self._hdfInstance._namespace}, name={self._hdfInstance._idx_name}")
            raise

        return self._vector_idxParams

    async def SetIdxParams(self) -> None:
        if self._vector_idxParams is None or self._vector_idxhealerScheduler is None:
            return

        self._logger.info(f"HealerOptions.SetIdxParams {self}")

        try:
            await self._vector_asyncClient.index_update(namespace=self._hdfInstance._namespace,
                                                        name=self._hdfInstance._idx_name,
                                                        hnsw_update_params=vectorTypes.HnswIndexUpdate(healer_params=vectorTypes.HnswHealerParams(schedule=self._vector_idxhealerScheduler)),
                                                        timeout=4)
            await asyncio.sleep(self._delaytimesecs) #need to wait to take effect

            if self._logger.level == logging.DEBUG:
                params = await self._vector_asyncClient.index_get(namespace=self._hdfInstance._namespace,
                                                                    name=self._hdfInstance._idx_name,
                                                                    timeout=4)
                self._logger.debug(f"HealerOptions.SetIdxParams Changed to '{'None' if params is None else params.hnsw_params.healer_params.schedule}'")

        except Exception as e:
            self._logger.exception(f"SetIdxParams.SetIdxParams failed ns={self._hdfInstance._namespace}, name={self._hdfInstance._idx_name}")
            raise

    async def ChangeIdxHealerSchedule(self, newInternalSecs:int) -> Optional[str]:
        '''
        newInternalSecs --  0  - disable healer
                            -1 - Now
                            >0 - Number of seconds interval
                            None or -2 - No operation
        '''

        if self._vector_idxParams is None or newInternalSecs is None:
            return

        self._vector_idxhealerScheduler = HealerOptions._determineschedulerstr(newInternalSecs)

        if self._vector_idxhealerScheduler is None:
            return None

        self._logger.info(f"HealerOptions.ChangeIdxHealerSchedule {self} ({newInternalSecs})")

        try:
            await self._vector_asyncClient.index_update(namespace=self._hdfInstance._namespace,
                                                        name=self._hdfInstance._idx_name,
                                                        hnsw_update_params=vectorTypes.HnswIndexUpdate(healer_params=vectorTypes.HnswHealerParams(schedule=self._vector_idxhealerScheduler)),
                                                        timeout=4)
            await asyncio.sleep(self._delaytimesecs) #need to wait to take effect

            if self._logger.level == logging.DEBUG:
                params = await self._vector_asyncClient.index_get(namespace=self._hdfInstance._namespace,
                                                                    name=self._hdfInstance._idx_name,
                                                                    timeout=4)
                self._logger.debug(f"HealerOptions.ChangeIdxHealerSchedule Changed to '{'None' if params is None else params.hnsw_params.healer_params.schedule}'")

        except Exception as e:
            self._logger.exception(f"SetIdxParams.ChangeIdxHealerSchedule failed ns={self._hdfInstance._namespace}, name={self._hdfInstance._idx_name}")
            raise
        return self._vector_idxhealerScheduler

    def __str__(self):
        if self._hdfInstance is None:
            return 'HealerOptions()'
        scheduleParams:str = HealerOptions.Determine_State(self._schedulerSecs)

        idxParams:str = 'None'
        if self._vector_idxParams is not None:
            idxParams = self._vector_idxParams.hnsw_params.healer_params.schedule

        return f"HealerOptions(ns={self._hdfInstance._namespace},idxname={self._hdfInstance._idx_name},action={scheduleParams},orginalIdxHealer='{idxParams}',idxHealer='{self._vector_idxhealerScheduler}')"