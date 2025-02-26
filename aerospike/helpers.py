
from aerospike_vector_search import types as vectorTypes

def set_hnsw_params_attrs(__obj :object, __dict: dict) -> object:
    for key in __dict:
        if key == 'batching_params':
            setattr(
                __obj,
                key,
                set_hnsw_params_attrs(
                        vectorTypes.HnswBatchingParams(),
                        __dict[key].__dict__,
                )
            )
        elif key == 'index_caching_params' or key == 'caching_params':
            setattr(
                __obj,
                key,
                set_hnsw_params_attrs(
                        vectorTypes.HnswCachingParams(),
                        __dict[key].__dict__,
                )
            )
        elif key == 'healer_params':
            setattr(
                __obj,
                key,
                set_hnsw_params_attrs(
                        vectorTypes.HnswHealerParams(),
                        __dict[key].__dict__,
                )
            )
        elif key == 'merge_params':
            setattr(
                __obj,
                key,
                set_hnsw_params_attrs(
                        vectorTypes.HnswIndexMergeParams(),
                        __dict[key].__dict__,
                )
            )
        elif key == 'record_caching_params':
            setattr(
                __obj,
                key,
                set_hnsw_params_attrs(
                        vectorTypes.HnswCachingParams(),
                        __dict[key].__dict__,
                )
            )
        elif (type(__dict[key]) is str
                and (__dict[key].lower() == "none"
                    or __dict[key].lower() == "null")):
            setattr(__obj, key, None)
        else:
            setattr(__obj, key, __dict[key])
    return __obj

def hnswstr(hnswparams : vectorTypes.HnswParams) -> str:
        if hnswparams is None:
            return ''
        if hnswparams.batching_params is None:
            batchingparams = ''
        else:
            batchingparams = f"maxrecs:{hnswparams.batching_params.max_index_records}, max_reindex:{hnswparams.batching_params.max_reindex_records}, interval:{hnswparams.batching_params.index_interval}"
        if hnswparams.record_caching_params is None:
            cachingparams = ''
        else:
            cachingparams = f"max_entries:{hnswparams.record_caching_params.max_entries}, expiry:{hnswparams.record_caching_params.expiry}"
        if hnswparams.index_caching_params is None:
            cachingIdxparams = ''
        else:
            cachingIdxparams = f"max_entries:{hnswparams.index_caching_params.max_entries}, expiry:{hnswparams.index_caching_params.expiry}"
        if hnswparams.healer_params is None:
            healerparams = ''
        else:
            healerparams = f"max_scan_rate_per_node:{hnswparams.healer_params.max_scan_rate_per_node}, max_scan_page_size:{hnswparams.healer_params.max_scan_page_size}, re_index_percent: {hnswparams.healer_params.re_index_percent}, schedule: {hnswparams.healer_params.schedule}, parallelism: {hnswparams.healer_params.parallelism}"
        if hnswparams.merge_params is None:
            mergeparams = ''
        else:
            mergeparams = f"index_parallelism: {hnswparams.merge_params.index_parallelism}, reindex_parallelism:{hnswparams.merge_params.reindex_parallelism}"

        return f"m:{hnswparams.m}, efconst:{hnswparams.ef_construction}, ef:{hnswparams.ef}, batching:{{{batchingparams}}}, recordcaching:{{{cachingparams}}}, indexcaching:{{{cachingIdxparams}}} healer:{{{healerparams}}}, merge:{{{mergeparams}}}"
