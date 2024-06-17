# hdf_import.py

This module will import a generated ANN HDF dataset. If the dataset doesn’t locally exist, it will be imported from ANN repro. All datasets are cached in the “./data” folder.

You can obtain a list of arguments by running:

```
python hdf_import.py –help
```

Below is a review of the argument:

\-h, --help

Show this help message and exit

\-d DS, --dataset DS

The ANN dataset (DS) to load training points from (default: glove-100-angular)

\-c N, --concurrency N

The maximum number of concurrent tasks (N) used to population the index.

“N’ Values are:

-   \< 0 – All records are upserted, concurrently waiting for the upsert confirmation once all upserts are submitted
-   0 -- Disable Population  
    If the index doesn’t existence, it is created. The “wait for index completion” is still being performed.
-   1 -- One record is upserted and confirmed at a time (sync)
-   \> 1 -- The number of records upserted and confirmed, concurrently (async)

    After population occurs, the module will go into “wait for index completion” mode by default. When this occurs, the module will wait until all populated index records are merged into the Aerospike DB.

    (default: 500)

\--idxdrop

If the Vector Index existence, it will be dropped. Otherwise, it is updated. (default: False)

\--idxnowait

Waiting for index completion is disabled. The module will continue without waiting for the index records to  
be merged into the Aerospike DB (default: False)

\-E EVT, --exhaustedevt EVT

This determines how the Resource Exhausted event is handled.   
This event occurs with the Vector Server merge queue is filled and cannot process any additional  
population requests.

“EVT” Values are:

-   \< 0 – All population events are stopped and will not resume until the index merger queue is cleared. This is done by “waiting for index completion” to occur. Once the queue is cleared, the population will be restarted.
-   0 -- Disable event handling (just re-throws the exception)
-   \>= 1 – All population events are stopped, and the module will wait for “EVT” seconds. Once the interval is reached, the population will be restarted.

(default: -1)

\-m RECS, --maxrecs RECS

Determines the maximum number of records to populated. A value of -1 (default) all records in  
the HDF dataset are populated.

(default: -1)

\-p port, --vectorport port

The Vector Server Port (default: 5000)

\-a HOST, --host HOST

The Vector Server’s IP Address or Host Name (default: localhost)

\-A HOST:PORT [HOST:PORT ...], --hosts HOST:PORT [HOST:PORT ...]

A list of host and optional port. Each pair is separated by a space.  
Example: 'hosta:5000' or 'hostb' (default: [localhost:5000])

If provided, each population request is distributed over the list of hosts.

Note: if this is provided, “—host” and “—port” arguments are ignored.

\-l, --vectorloadbalancer

Use Vector's Load Balancer.

Note: if “—hosts” argument is used, only the first host in the list is used (reminding hosts are ignored)

(default: False)

\-T, --vectortls

Use TLS to connect to the Vector Server

(default: False)

\-n NS, --namespace NS

The Aerospike Namespace  
(default: test)

\-N NS, --idxnamespace NS

Aerospike Namespace where the vector index will be located.  
Defaults to the value of “—namespace”.

\-s SET, --setname SET

The Aerospike Set Name  
(default: HDF-data)

\-I IDX, --idxname IDX

The Vector Index Name.  
Defaults to the Set Name (--setname) with the suffix of '_idx'

\-g, --generatedetailsetname

Generates a unique Set Name (--setname) based on distance type, dimensions, index params, etc.   
(default: False)

\-b BIN, --vectorbinname BIN

The Aerospike Bin Name where the vector is stored  
(default: “HDF_embedding”)

\-D DIST, --distancetype DIST

The Vector's Index Distance Type as defined by the Vector Phyton API.  
The default is to select the index type based on the ANN dataset

\-P PARM, --indexparams PARM

The Vector's Index Params (HnswParams) as defined by the Vector Phyton API.  
(default: {"m": 16, "ef_construction": 100, "ef": 100})

\-L LOG, --logfile LOG

The logging file path, if provided.   
 The default is to stdout.

\--loglevel LEVEL

The Logging level (default: INFO)

\--driverloglevel DLEVEL

The Vector Phyton Driver's Logging level (default: NOTSET)

\--prometheus PORT

The Prometheus Port (default: 9464)

\--prometheushb SECS

Prometheus heartbeat in secs. The heartbeat updates common information to Prometheus  
(default: 5 seconds)

\--exitdelay wait

Upon exist, the module will sleep ensuring all Prometheus events are captured  
(default: 20)

# hdf_query.py

This module will query using the ANN neighbor query vector defined in the ANN dataset that was downloaded and populated using [hdf_import.py](#hdf_importpy).

You can obtain a list of arguments by running:

```
python hdf_import.py –help
```

Below is a review of the argument:

\-h, --help

Show this help message and exit

\-d DS, --dataset DS

The ANN dataset (DS) to load training points from (default: glove-100-angular)

\-r RUNS, --runs RUNS

The number of times the query requests will run based on the ANN dataset.  
For example: If the ANN dataset request 1,000 queries and if this value is 10; The total number of query requests will be 10,000 (1,000 \* 10).  
(default: 10)

\--limit NEEIGHBORS

The number of neighbors to return from each query request  
(default: 100)

\--parallel

Each “run” is conducted concurrently  
(default: False)

\--check

Each query result is checked to determine if the result is correct.  
 {default False)

\-p port, --vectorport port

The Vector Server Port (default: 5000)

\-a HOST, --host HOST

The Vector Server’s IP Address or Host Name (default: localhost)

\-A HOST:PORT [HOST:PORT ...], --hosts HOST:PORT [HOST:PORT ...]

A list of host and optional port. Each pair is separated by a space.  
Example: 'hosta:5000' or 'hostb' (default: [localhost:5000])

If provided, each query request is distributed over the list of hosts.

Note: if this is provided, “—host” and “—port” arguments are ignored.

\-l, --vectorloadbalancer

Use Vector's Load Balancer.

Note: if “—hosts” argument is used, only the first host in the list is used (reminding hosts are ignored)

(default: False)

\-T, --vectortls

Use TLS to connect to the Vector Server

(default: False)

\-N NS, --idxnamespace NS

Aerospike Namespace where the vector index will be located.  
Defaults to the value of “—namespace”.

\-s SET, --setname SET

The Aerospike Set Name  
(default: HDF-data)

\-I IDX, --idxname IDX

The Vector Index Name.  
Defaults to the Set Name (--setname) with the suffix of '_idx'

\-g, --generatedetailsetname

Generates a unique Set Name (--setname) based on distance type, dimensions, index params, etc.   
(default: False)

\-S PARM, --searchparams PARM

The Vector's Search Params (HnswParams) as defined by the Vector Phyton API.  
 Defaults to --indexparams

\-L LOG, --logfile LOG

The logging file path, if provided.   
 The default is to stdout.

\--loglevel LEVEL

The Logging level (default: INFO)

\--driverloglevel DLEVEL

The Vector Phyton Driver's Logging level (default: NOTSET)

\--prometheus PORT

The Prometheus Port (default: 9464)

\--prometheushb SECS

Prometheus heartbeat in secs. The heartbeat updates common information to Prometheus  
(default: 5 seconds)

\--exitdelay wait

Upon exist, the module will sleep ensuring all Prometheus events are captured  
(default: 20)

# Prometheus

The module outputs certain meters to Prometheus. They are:

-   `aerospike.hdf.heartbeat
    This event is defined as a gauge. It provides information about the status of the module including the following attributes:`
    -   `"ns” – Aerospike Namespace`
    -   `"set” – Aerospike Set                                                   `
    -   `"idxns” – Aerospike Vector Index Namespace`
    -   `"idx" – The Aerospike Vector Index Name                                                     `
    -   `"idxbin" – The Vector’s Bin Name`
    -   `"idxdist" – The Vector’s API Distance Type`
    -   `"dims" – Vector’s dimensions`
    -   `"poprecs" – The number of records in the ANN dataset that will be populated`
    -   `"querynbrlmt" – The number of neighbors returned in a query                                                       `
    -   `"dataset” – The ANN dataset`
    -   `"paused" – True if the population because of an event like “resource exhausted”`
    -   `"action” – If importing (populating) or querying`
-   `aerospike.hdf.populate
    Current record rate that have been upserted. Defined as a counter.
    Attributes:`
    -   `"type" -- upsert`
    -   `"ns" -- Namespace`
    -   `"set" – Set Name`
-   `aerospike.hdf.query
    Current query rate. Defined as a counter.
    Attributes:`
    -   `"ns" -- Namespace`
    -   `"` `idx" – Index Name`
-   `aerospike.hdf.exception
    Current exception rate. Defined as a counter.
    Attributes:`
    -   `"exception_type" – Type of exception`
    -   `"handled_by_user" – if handled by user code`
    -   `"ns" -- Namespace`
    -   `"set" – Set`
    -   `“idx” – Index name`
-   `aerospike.hdf.waitidxcompletion
    Current number of waiting for index merge completions being conducted. Defined as a counter.
    Attributes:`
    -   `"ns" – Index Namespace`
    -   `"idx" – Index Name`
-   `aerospike.hdf.dropidxtime
    The amount of time to perform an index drop. Defined as a histogram.
    Attributes:`
    -   `"ns" – Index Namepsace`
    -   `"idx" – Index Name`
-   `aerospike.hdf.populate.recs
    The current number of records upserted. Defaulted as a gauge.
    Attributes:`
    -   `"ns" -- Namespace`
    -   `"set" – Set Name`
-   `aerospike.hdf.query.recs
    The current number of queries performed. Defined as a gauge.
    Attributes:`
    -   `"ns" – Index Namespace`
    -   `"idx" – Index Name`
-   `aerospike.hdf.query.runs
    The current number of runs for a query. Defined as a gauge.
    Attributes:`
    -   `"ns" – Index Namespace`
    -   `"idx" – Index Name`
