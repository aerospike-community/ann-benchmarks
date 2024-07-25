# Backoff Logic when encountering Resource Exhausted

# The “back-off” logic is as follows:

-   When the exception is received:
    -   First record in error will perform the following actions:
        -   Signal the “main” populating task to go into “sleep mode” so that additional records will not be upserted.
        -   A warning message is logged.
    -   All records in error will do the following:
        -   Call “wait for index completion.”
        -   Once the index is built the following occurs:
            -   Re-upsert the error records
            -   If successful, signal the “main” populating task to re-start populating.
            -   A warning message is logged, stating population has re-started.

# Environmental Variables

Below are the Environmental Variables:

-   AVS_LOGLEVEL -- The Vector Client API's Log Level. Defaults "WARNING"
    -   Possible Values:
    -   CRITICAL
    -   FATAL
    -   ERROR
    -   WARNING
    -   WARN
    -   INFO
    -   DEBUG
    -   NOTSET

        Note: The logging file is determined by "APP_LOGFILE"

-   AVS_HOST -- The AVS Server's Address. Defaults to "localhost"
-   AVS_PORT -- The AVS Server's Port. Defaults to 5000
-   AVS_VERIFY_TLS -- A Boolean value to determine if TLS connection is required. Defaults to True.
-   AVS_USELOADBALANCER -- The AVS Server's Address is a Load Balancer. Default False.
-   AVS_NAMESPACE -- The Vector's Namespace. Defaults to "test"
-   AVS_SET -- The Vector's Set name. Defaults to "ANN-data"

    This behavior is determined by the "uniqueSetIdxName" argument defined in the config.yml file.

    The default (True) behavior is to create a unique Set name where this is the prefix to that name.

The name has the following parts:

```
{AVS_SET}_{ANN Distance Type}_{AVS Idx Type}_{Dimension}_{hnsw m}_{hnsw ef construction}_{hnsw ef}
```

Example:

ANN-data_angular_COSINE_20_16_100_100

If "uniqueSetIdxName" is false, the Set name is as follows:

```
{AVS_SET}__{ANN Distance Type}_{AVS Idx Type}
```

Example:

ANN-data_angular_COSINE

-   APP_LOGFILE -- The Aerospike's ANN Logging file. Default is "AerospikeANN.log".

    The folder is always the current working directory.

-   APP_LOGLEVEL -- The Aerospike's ANN Log Level. Defaults "INFO"
    -   Possible Values:
    -   CRITICAL
    -   FATAL
    -   ERROR
    -   WARNING
    -   WARN
    -   INFO
    -   DEBUG
    -   NOTSET

Note: For performance testing this should be set to "NOTSET".

When running in a docker container, logging is disabled.

-   APP_DROP_IDX -- A Boolean value that will determine if the Vector index is dropped if it already exists. The default is to use "dropIdx" argument in the config.yml file.
-   APP_INDEX_SLEEP -- The amount of time to sleep after the index is dropped. The default is 0.

    Possible values are:

    -   0 -- Don't Sleep
    -   \< 0 -- The number of seconds to sleep
-   APP_POPULATE_TASKS -- The number of concurrent records upserted (put) tasks that are performed during the index population phase. When this number of records are upserted, the app will wait until all upserts are completed and then process the next set of records. The default is 5000.

    Values:

    -   \< 0 -- All records are upserted, concurrently, and the app will only wait for the upsert completion before waiting for index completion.
    -   0 or 1 -- One record is upserted at a time (sync)
    -   \> 1 -- The number of records upserted, concurrently (async), before the app waits for the upserts to complete.
-   APP_PINGAVS -- Checks to determine if the AVS server is reachable via ping. Default is False.
-   APP_CHECKRESULT -- Checks the Vector Search results for failed results or Zero Distance. Default is True

    Note: This value is always false if running in a docker container.

    This should be set to False when conducting performance testing!

The default bin name for the vectors is always "ANN_embedding".

# config.yml file

Using the config.yml file. The Aerospike ANN config.yml file can support the different ANN run group configurations. It is suggested that the Aerospike ANN application is ran using the ANN Distance Type configuration.

Using this configuration, we can match each ANN distance type (i.e., Angular, Euclidean, Jaccard, etc.) to the "best" Aerospike Vector index type. Below is an example of this configuration with comments regarding the behavior of each parameter:

```
float:
#This defines a run group based on the ANN angular datasets.
  angular:
#All entries to “run_groups” keyword are required as-is (cannot change the values or structure)!
  - base_args: ['@metric',  '@dimension']
    constructor: Aerospike
    disabled: false #can change to true to disable this run-group
    docker_tag: ann-benchmarks-aerospike
    module: ann_benchmarks.algorithms.aerospike
    name: aerospike
    run_groups:
      cosine: #Should match Idx Type
#This grouping is reqired
        args: [
          [cosine], #Idx Type, any Aerospike Index Type, case insensitive). This is required…
#A collection of HnswParams where each param is ran as a separate ran for this Idx Type. This is required and must have at least one item.
          [{m: 8, ef_construction: 64, ef: 8},
            {m: 16, ef_construction: 128, ef: 8} ],
#Unique Set/Index Name (optional, default True). See the “AVS_SET” environment variable above.
          [True], 
#True to Drop Idx and Re-Populate, optional default true. See “APP_DROP_IDX” environment variable above.
          [True],
#Determines what phases are executed. Values are:
#	IdxPopulateOnly – only conduct the populate index phase,
#	QueryOnly – only perform the vector search phase,
#	AllOps – All phases (optional default value)
          [AllOps] 
        ]
#This grouping is required
        query_args: [ 
# If provided (optional), overrides the HnswParams defined above for the vector search phase
          [null, #Uses default defined above
            {ef: 10} #Override “ef” above
		]
      ]
#This defines another run group based on the ANN Euclidean datasets.
#This show using the required params.
  euclidean:
  - base_args: ['@metric',  '@dimension']
    constructor: Aerospike
    disabled: false
    docker_tag: ann-benchmarks-aerospike
    module: ann_benchmarks.algorithms.aerospike
    name: aerospike
    run_groups:
      SQUARED_EUCLIDEAN:
              args: [
                [SQUARED_EUCLIDEAN], #Idx Type
                [{m: 16, ef_construction: 100, ef: 100}]
              ]      
              query_args: [ 
                []  
              ]
```
